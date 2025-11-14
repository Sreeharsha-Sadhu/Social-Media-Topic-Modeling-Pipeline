"""
stage4_global_analysis.py

Stage 4: Global Topic Analysis (Dual-mode: Pandas or Spark) + Live feed integration (Option C)

Features:
 - Synthetic/batch topic modeling (unchanged behavior)
 - Live feed analysis per-user (reddit/linkedin/tweet) with incremental fetch:
     - Get posts since last run OR top_n recent posts, whichever is fewer
     - If posts > 50 -> full embeddings + KMeans topics + summaries
     - If posts <= 50 -> lightweight single-LLM summary (fast)
 - Persists results to:
     - global_topics (topic_id, topic_title, summary_text, sample_posts)
     - post_topic_mapping (post_id, topic_id)
     - live_runs (user_id, source, last_run_at, post_count, summary_cache_id)
     - live_outputs (id, user_id, source, created_at, payload_json)
 - Provides `run_topic_model_on_text_list` for reuse by live analyzers.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql import functions as F
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans as SkKMeans
from tqdm import tqdm
from transformers import pipeline

from src.core import config, utils
from src.core.logging_config import get_logger

logger = get_logger(__name__)

# Local cache for heavy models
_model_cache: Dict[str, Any] = {}

# DB sample joiner
SAMPLE_JOINER = " ||| "

# ---------------------------------------------------------------------
# DB: Additional tables for live runs
# ---------------------------------------------------------------------
LIVE_SCHEMA_SQL = """
                  -- persistent cache for live runs (non-destructive to other tables)
                  CREATE TABLE IF NOT EXISTS live_outputs
                  (
                      id           SERIAL PRIMARY KEY,
                      user_id      TEXT,
                      source       TEXT,
                      created_at   TIMESTAMP DEFAULT now(),
                      payload_json JSONB
                  );

                  CREATE TABLE IF NOT EXISTS live_runs
                  (
                      user_id     TEXT,
                      source      TEXT,
                      last_run_at TIMESTAMP,
                      post_count  INTEGER,
                      output_id   INTEGER REFERENCES live_outputs (id),
                      PRIMARY KEY (user_id, source)
                  );

                  CREATE TABLE IF NOT EXISTS live_topics
                  (
                      id           SERIAL PRIMARY KEY,
                      user_id      TEXT NOT NULL,
                      source       TEXT NOT NULL,
                      topic_title  TEXT,
                      summary_text TEXT,
                      sample_posts TEXT,
                      created_at   TIMESTAMP DEFAULT NOW()
                  );

                  CREATE TABLE IF NOT EXISTS live_post_topic_mapping
                  (
                      id         SERIAL PRIMARY KEY,
                      user_id    TEXT NOT NULL,
                      source     TEXT NOT NULL,
                      post_ref   TEXT,
                      topic_ref  INTEGER REFERENCES live_topics (id) ON DELETE CASCADE,
                      created_at TIMESTAMP DEFAULT NOW()
                  );
                  
                  """


# ---------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------
def get_llm_pipeline():
    """Load and cache the FLAN-T5-Large pipeline used for summarization/title."""
    if "llm" not in _model_cache:
        logger.info("Loading FLAN-T5-Large model (this may take a while)...")
        model_name = "google/flan-t5-large"
        if not torch.cuda.is_available():
            logger.warning("CUDA not available; LLM will run on CPU (slow).")
            _model_cache["llm"] = pipeline("text2text-generation", model=model_name)
        else:
            _model_cache["llm"] = pipeline(
                "text2text-generation",
                model=model_name,
                device=0,
                dtype=torch.float16
            )
        logger.info("LLM ready.")
    return _model_cache["llm"]


def get_sentence_transformer():
    """Load/cached SentenceTransformer model for embeddings."""
    if "sbert" not in _model_cache:
        logger.info("Loading SentenceTransformer 'all-MiniLM-L6-v2' ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_cache["sbert"] = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        logger.info("SentenceTransformer ready.")
    return _model_cache["sbert"]


# ---------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------
def restore_readable_text(text: str) -> str:
    """Recover spacing after light normalization (reverse of earlier cleaning)."""
    if not text:
        return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-z])', r'\1 \2', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def prepare_llm_text(texts: List[str]) -> str:
    """Join texts into an LLM-friendly paragraph (makes sentences readable)."""
    cleaned = []
    for t in texts:
        if not t:
            continue
        t = t.strip()
        if not re.search(r'[.!?]$', t):
            t = t + "."
        cleaned.append(t.capitalize())
    return " ".join(cleaned)


def summarize_topic_text(generator, full_text: str) -> str:
    """Chunk then summarize long text, then meta-summarize."""
    if not full_text:
        return "(No posts in this topic.)"
    
    chunk_size = 3000
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    partials = []
    for chunk in chunks:
        prompt = (
            "Summarize this discussion into a concise paragraph capturing the theme and sentiment. "
            "Avoid repetition.\n\n"
            f"{chunk}\n\nSummary:"
        )
        try:
            res = generator(
                prompt,
                truncation=True,
                max_new_tokens=120,
                do_sample=False,
                no_repeat_ngram_size=3
            )
            text = res[0].get("generated_text") or res[0].get("text") or ""
            if text:
                partials.append(text.strip())
        except Exception as e:
            logger.warning("Chunk summary failed: %s", e)
    if not partials:
        return "(No coherent summary generated.)"
    joined = " ".join(partials)
    final_prompt = (
        "Combine the following short summaries into one cohesive paragraph describing the main themes:\n\n"
        f"{joined}\n\nFinal summary:"
    )
    try:
        out = generator(final_prompt, truncation=True, max_new_tokens=180, do_sample=False)
        text = out[0].get("generated_text") or out[0].get("text") or ""
        return text.strip() if text else " ".join(partials)[:1200]
    except Exception:
        return " ".join(partials)[:1200]


def generate_topic_title(generator, sample_posts: List[str], summary_text: str) -> str:
    """Return a short 3-7 word title for the topic (best-effort)."""
    try:
        joined = " ".join(sample_posts)[:700]
        prompt = (
            "Create a concise 3-7 word title (no punctuation) summarizing these social posts and summary.\n\n"
            f"Posts: {joined}\nSummary: {summary_text}\n\nTitle:"
        )
        out = generator(prompt, truncation=True, max_new_tokens=12, do_sample=False)
        t = out[0].get("generated_text") or out[0].get("text") or ""
        title = re.sub(r'[^A-Za-z0-9\s-]', '', t).strip()
        if not title:
            return "Untitled Topic"
        words = title.split()
        return " ".join(words[:7])
    except Exception:
        return "Untitled Topic"


# ---------------------------------------------------------------------
# Core reusable function for topic modeling on a list of texts
# ---------------------------------------------------------------------
def run_topic_model_on_text_list(
        texts: List[str],
        num_topics: int = 6,
        min_cluster_for_full_pipeline: int = 2,
        use_spark: bool = False,
) -> Dict[str, Any]:
    """
    Run topic modeling on an in-memory list of texts.

    Returns a dict:
      {
        "type": "clustered" | "single_summary",
        "topics": [
            {"topic_id": int, "title": str, "summary": str, "sample_posts": [str], "post_ids": [str]}
        ],
        "mappings": pd.DataFrame or list-of-dicts (post_id->topic_id),
        "raw_texts": [...],
    }
    """
    # Basic sanity
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return {"type": "empty", "topics": [], "mappings": [], "raw_texts": []}
    
    # restore readability
    restored = [restore_readable_text(t) for t in texts]
    generator = get_llm_pipeline()
    
    # Hybrid policy (we treat the number of texts later)
    n = len(restored)
    result: Dict[str, Any] = {"raw_texts": restored, "topics": []}
    
    # If small workload -> single LLM summary (fast)
    if n <= 50:
        logger.info("Using lightweight LLM summarization for %d posts.", n)
        blob = prepare_llm_text(restored)[:4000]
        summary = summarize_topic_text(generator, blob)
        title = generate_topic_title(generator, restored[:3], summary)
        result.update({
            "type": "single_summary",
            "topics": [
                {"topic_id": 0, "title": title, "summary": summary, "sample_posts": restored[:3],
                 "post_ids": list(range(n))}
            ],
            "mappings": [{"post_index": i, "topic_id": 0} for i in range(n)]
        })
        return result
    
    # Otherwise run full pipeline (embeddings + KMeans)
    logger.info("Running full pipeline: embeddings + KMeans for %d posts.", n)
    sbert = get_sentence_transformer()
    embeddings = sbert.encode(restored, batch_size=32, show_progress_bar=True)
    np_emb = np.array(embeddings)
    
    # clamp num_topics <= n
    num_topics = max(2, min(num_topics, n))
    try:
        # try sklearn by default (driver)
        km = SkKMeans(n_clusters=num_topics, random_state=42, n_init=10)
        preds = km.fit_predict(np_emb)
    except Exception as e:
        logger.warning("Sklearn KMeans failed: %s", e)
        # if spark available and requested, fallback is handled by caller
        preds = np.zeros(len(restored), dtype=int)
    
    # group posts by cluster
    topics = []
    for tid in range(int(num_topics)):
        indices = [i for i, p in enumerate(preds) if int(p) == tid]
        if not indices:
            continue
        posts_for_topic = [restored[i] for i in indices]
        unique_posts = list(dict.fromkeys(posts_for_topic))
        full_text = prepare_llm_text(unique_posts)[:3500]
        try:
            summary = summarize_topic_text(generator, full_text)
            title = generate_topic_title(generator, unique_posts[:3], summary)
        except Exception as e:
            logger.warning("LLM summarization failed for topic %d: %s", tid, e)
            summary = "Error generating summary."
            title = "Untitled Topic"
        
        topics.append({
            "topic_id": int(tid),
            "title": title,
            "summary": summary,
            "sample_posts": unique_posts[:3],
            "post_indices": indices
        })
    
    mappings = [{"post_index": i, "topic_id": int(preds[i])} for i in range(len(preds))]
    
    result.update({"type": "clustered", "topics": topics, "mappings": mappings})
    return result


# ---------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------
def _ensure_live_tables():
    """Create live_outputs and live_runs if not present."""
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(LIVE_SCHEMA_SQL)
                conn.commit()
        logger.debug("Ensured live tables exist.")
    except Exception as e:
        logger.warning("Could not ensure live tables: %s", e)


def _save_live_output(user_id: str, source: str, payload: Dict[str, Any]) -> int:
    """Persist a live run payload JSON into live_outputs and return id."""
    try:
        payload_json = json.dumps(payload)
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO live_outputs (user_id, source, payload_json) VALUES (%s, %s, %s) RETURNING id",
                    (user_id, source, payload_json)
                )
                out_id = cur.fetchone()[0]
                conn.commit()
        logger.debug("Saved live output id=%s for %s/%s", out_id, user_id, source)
        return int(out_id)
    except Exception as e:
        logger.error("Failed to save live output: %s", e)
        raise


def _upsert_live_run(user_id: str, source: str, post_count: int, output_id: int):
    """Upsert last_run_at and reference to live_outputs row."""
    try:
        now = datetime.now(timezone.utc)
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO live_runs (user_id, source, last_run_at, post_count, output_id)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, source) DO UPDATE
                        SET last_run_at = EXCLUDED.last_run_at,
                            post_count  = EXCLUDED.post_count,
                            output_id   = EXCLUDED.output_id
                    """,
                    (user_id, source, now, post_count, output_id)
                )
                conn.commit()
        logger.debug("Upserted live_runs for %s/%s", user_id, source)
    except Exception as e:
        logger.error("Failed to upsert live_runs: %s", e)


def get_last_run_for_user(user_id: str, source: str) -> Optional[Dict[str, Any]]:
    """Return row (last_run_at, post_count, output_id) or None."""
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT last_run_at, post_count, output_id FROM live_runs WHERE user_id = %s AND source = %s",
                    (user_id, source)
                )
                row = cur.fetchone()
        if not row:
            return None
        return {"last_run_at": row[0], "post_count": row[1], "output_id": row[2]}
    except Exception as e:
        logger.warning("Failed to read last_run_for_user: %s", e)
        return None


def _load_live_output_by_id(output_id: int) -> Optional[Dict[str, Any]]:
    """Load saved payload by id (returns parsed JSON)."""
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload_json FROM live_outputs WHERE id = %s", (output_id,))
                row = cur.fetchone()
        if not row:
            return None
        return json.loads(row[0])
    except Exception as e:
        logger.warning("Failed to load live output %s: %s", output_id, e)
        return None


# ---------------------------------------------------------------------
# Live-run entrypoint
# ---------------------------------------------------------------------
def run_live_analysis_for_user(
        user_id: str,
        source: str = "reddit",
        fetcher_callable=None,
        top_n: int = 100,
        prefer_since_last_run: bool = True,
        full_pipeline_threshold: int = 50,
        num_topics: int = 6
) -> Dict[str, Any]:
    """
    Orchestrate a live analysis for a user.

    Args:
      - user_id: user identifier in your system
      - source: 'reddit' | 'linkedin' | 'twitter' (informational)
      - fetcher_callable: function that accepts (user_id, since:Optional[datetime], limit:int) and returns list of posts
          Each post should be a dict with at least: {'post_id': str, 'text': str, 'created_at': iso str}
      - top_n: maximum posts to consider
      - prefer_since_last_run: if True, attempt incremental (since last run) fetch
      - full_pipeline_threshold: if <= threshold -> lightweight summary, else full pipeline
      - num_topics: desired number of clusters for full pipeline
    Returns:
      - dict containing result and metadata. On error, will return cached last output if available.
    """
    logger.info("Starting live analysis for %s/%s (top_n=%d)", user_id, source, top_n)
    _ensure_live_tables()
    
    # Validate fetcher existence
    if not callable(fetcher_callable):
        raise ValueError("fetcher_callable must be provided and callable(user_id, since, limit)")
    
    # Determine 'since' from last run
    last = get_last_run_for_user(user_id, source)
    since = last["last_run_at"] if (last and prefer_since_last_run and last.get("last_run_at")) else None
    
    try:
        # Fetch new posts: try since, but ensure limit top_n
        posts = fetcher_callable(user_id, since=since, limit=top_n)
        if not posts:
            # fallback: if nothing returned and we have previous output -> return cached
            logger.warning("No posts returned by fetcher for %s/%s. Returning cached output if available.", user_id,
                           source)
            if last and last.get("output_id"):
                cached = _load_live_output_by_id(last["output_id"])
                return {"status": "cached_fallback", "payload": cached}
            return {"status": "no_data", "payload": None}
        
        # Restrict to the most recent N (whichever is fewer: since-run results or top_n)
        # fetcher_callable should respect 'limit', but ensure correct ordering:
        posts_sorted = sorted(posts, key=lambda p: p.get("created_at", ""), reverse=True)
        selected = posts_sorted[:top_n]
        
        # Extract texts (and preserve post ids)
        texts = [p.get("text") or p.get("content") or "" for p in selected]
        post_ids = [p.get("post_id") or p.get("id") or str(i) for i, p in enumerate(selected)]
        
        processed_count = len(texts)
        logger.info("Fetched %d posts (processing %d) for %s/%s", len(posts), processed_count, user_id, source)
        
        # Select pipeline based on count
        if processed_count <= full_pipeline_threshold:
            logger.info("Using single-LLM summarization pipeline (<= %d posts).", full_pipeline_threshold)
            res = run_topic_model_on_text_list(texts, num_topics=1)
        else:
            logger.info("Using full clustered pipeline (> %d posts).", full_pipeline_threshold)
            res = run_topic_model_on_text_list(texts, num_topics=num_topics)
        
        # Prepare payload to persist
        payload = {
            "user_id": user_id,
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "post_count": processed_count,
            "result": res,
            "post_ids": post_ids
        }
        
        # Save live output and upsert run pointer
        output_id = _save_live_output(user_id, source, payload)
        _upsert_live_run(user_id, source, processed_count, output_id)
        
        # If we got clustered topics, also persist into global_topics/post_topic_mapping
        try:
            engine = utils.get_sqlalchemy_engine()
            if res.get("type") == "clustered":
                # prepare global_topics df
                rows = []
                for t in res["topics"]:
                    rows.append({
                        "topic_id": int(t["topic_id"]),
                        "topic_title": t["title"],
                        "summary_text": t["summary"],
                        "sample_posts": SAMPLE_JOINER.join(t["sample_posts"])
                    })
                if rows:
                    pd.DataFrame(rows).to_sql("global_topics", engine, if_exists="append", index=False)
                
                # prepare mapping
                mapping_rows = []
                for m in res["mappings"]:
                    # mapping contains post_index -> topic_id
                    mapping_rows.append({
                        "post_id": post_ids[int(m["post_index"])],
                        "topic_id": int(m["topic_id"])
                    })
                if mapping_rows:
                    pd.DataFrame(mapping_rows).to_sql("post_topic_mapping", engine, if_exists="append", index=False)
            else:
                # single_summary: write a single row into global_topics and mapping of all posts -> topic 0
                topic = res["topics"][0]
                df_topic = pd.DataFrame([{
                    "topic_id": 0,
                    "topic_title": topic["title"],
                    "summary_text": topic["summary"],
                    "sample_posts": SAMPLE_JOINER.join(topic["sample_posts"])
                }])
                df_topic.to_sql("global_topics", engine, if_exists="append", index=False)
                mapping_rows = [{"post_id": pid, "topic_id": 0} for pid in post_ids]
                pd.DataFrame(mapping_rows).to_sql("post_topic_mapping", engine, if_exists="append", index=False)
        except Exception as e:
            logger.warning("Failed to persist topics/mappings from live run: %s", e)
        finally:
            try:
                engine.dispose()
            except Exception:
                pass
        
        return {"status": "ok", "payload": payload}
    
    except Exception as e:
        logger.error("Live analysis failed for %s/%s: %s", user_id, source, e)
        # fallback to cached output if exists
        last = get_last_run_for_user(user_id, source)
        if last and last.get("output_id"):
            cached = _load_live_output_by_id(last["output_id"])
            return {"status": "error_with_cache", "error": str(e), "payload": cached}
        return {"status": "error", "error": str(e), "payload": None}


# ---------------------------------------------------------------------
# Synthetic (batch) analysis entrypoints (unchanged behavior)
# ---------------------------------------------------------------------
def _run_global_analysis_pandas(engine) -> bool:
    """Run the original Pandas-based analysis (keeps behavior intact)."""
    try:
        sql_query = """
                    SELECT post_id, cleaned_content
                    FROM posts
                    WHERE cleaned_content IS NOT NULL
                      AND cleaned_content != '' \
                    """
        pd_posts = pd.read_sql(sql_query, engine)
        if len(pd_posts) < 20:
            logger.warning("Not enough posts (%d) to analyze.", len(pd_posts))
            return False
        
        logger.info("Generating embeddings...")
        sbert = get_sentence_transformer()
        embeddings = sbert.encode(pd_posts["cleaned_content"].tolist(), batch_size=16, show_progress_bar=True)
        
        num_topics = getattr(config, "PANDAS_ANALYSIS_TOPICS", 20)
        logger.info("Clustering into %s topics", num_topics)
        km = SkKMeans(n_clusters=num_topics, random_state=42, n_init=10)
        pd_posts["topic_id"] = km.fit_predict(np.array(embeddings))
        
        generator = get_llm_pipeline()
        topic_rows = []
        for topic_id, group in tqdm(pd_posts.groupby("topic_id"), total=num_topics, desc="Summarizing Topics"):
            unique_posts = list(dict.fromkeys(group["cleaned_content"].tolist()))
            restored = [restore_readable_text(p) for p in unique_posts]
            full_text = prepare_llm_text(restored)[:3500]
            sample_posts = restored[:3]
            summary = summarize_topic_text(generator, full_text)
            title = generate_topic_title(generator, sample_posts, summary)
            topic_rows.append((int(topic_id), title, summary, SAMPLE_JOINER.join(sample_posts)))
        
        pd_summaries = pd.DataFrame(topic_rows, columns=["topic_id", "topic_title", "summary_text", "sample_posts"])
        pd_mappings = pd_posts[["post_id", "topic_id"]]
        
        # persist
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE global_topics, post_topic_mapping RESTART IDENTITY CASCADE;")
                conn.commit()
        pd_summaries.to_sql("global_topics", engine, if_exists="append", index=False)
        pd_mappings.to_sql("post_topic_mapping", engine, if_exists="append", index=False)
        logger.info("Synthetic analysis complete and saved to DB.")
        return True
    except Exception as e:
        logger.error("Pandas analysis failed: %s", e)
        return False
    finally:
        try:
            if "engine" in locals():
                engine.dispose()
        except Exception:
            pass


def _run_global_analysis_spark(engine) -> bool:
    """Spark distributed variant (keeps your earlier behavior)."""
    try:
        spark = utils.get_spark_session()
        df_posts = (
            spark.read.format("jdbc")
            .option("url", f"jdbc:postgresql://{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}")
            .option("dbtable", "posts")
            .option("user", config.DB_USER)
            .option("password", config.DB_PASS)
            .load()
            .filter(F.col("cleaned_content").isNotNull() & (F.col("cleaned_content") != ""))
        )
        post_count = df_posts.count()
        if post_count < 20:
            logger.warning("Not enough posts (%d) to analyze.", post_count)
            return False
        
        sbert = get_sentence_transformer()
        posts_list = [r.cleaned_content for r in df_posts.select("cleaned_content").collect()]
        embeddings = sbert.encode(posts_list, batch_size=32, show_progress_bar=True)
        np_emb = np.array(embeddings)
        
        num_topics = config.SPARK_ANALYSIS_TOPICS
        embed_df = spark.createDataFrame([(i, Vectors.dense(vec)) for i, vec in enumerate(np_emb)],
                                         ["index", "features"]).repartition(
            max(2, spark.sparkContext.defaultParallelism)).cache()
        try:
            kmeans = SparkKMeans(k=num_topics, seed=42, featuresCol="features", predictionCol="prediction", maxIter=20)
            model = kmeans.fit(embed_df)
            clustered = model.transform(embed_df).select("index", F.col("prediction").alias("topic_id"))
        except Exception:
            logger.warning("Spark KMeans failed; falling back to sklearn on driver.")
            np_emb2 = np.array([r.features.toArray() for r in embed_df.collect()])
            preds = SkKMeans(n_clusters=num_topics, random_state=42, n_init=10).fit_predict(np_emb2)
            clustered = spark.createDataFrame(pd.DataFrame({"index": range(len(preds)), "topic_id": preds}))
        
        posts_with_index = df_posts.withColumn("index", F.monotonically_increasing_id())
        joined_df = posts_with_index.join(clustered, on="index").select("post_id", "cleaned_content", "topic_id")
        
        generator = get_llm_pipeline()
        topic_rows = []
        for topic_id in tqdm(range(num_topics), desc="Summarizing Topics"):
            topic_texts = [restore_readable_text(r.cleaned_content) for r in
                           joined_df.filter(F.col("topic_id") == topic_id).select("cleaned_content").collect()]
            if not topic_texts:
                continue
            unique_posts = list(dict.fromkeys(topic_texts))
            full_text = prepare_llm_text(unique_posts)[:4000]
            sample_posts = unique_posts[:3]
            summary = summarize_topic_text(generator, full_text)
            title = generate_topic_title(generator, sample_posts, summary)
            topic_rows.append((int(topic_id), title, summary, SAMPLE_JOINER.join(sample_posts)))
        
        pd_summaries = pd.DataFrame(topic_rows, columns=["topic_id", "topic_title", "summary_text", "sample_posts"])
        pd_mappings = joined_df.select("post_id", "topic_id").toPandas()
        
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE global_topics, post_topic_mapping RESTART IDENTITY CASCADE;")
                conn.commit()
        
        pd_summaries.to_sql("global_topics", engine, if_exists="append", index=False)
        pd_mappings.to_sql("post_topic_mapping", engine, if_exists="append", index=False)
        
        logger.info("Spark analysis complete.")
        return True
    except Exception as e:
        logger.error("Spark analysis error: %s", e)
        return False
    finally:
        logger.info("Spark analysis finished.")


# ---------------------------------------------------------------------
# Setup DB tables for analysis
# ---------------------------------------------------------------------
def setup_database_tables():
    """Create or recreate the global analysis DB tables (destructive for analysis tables)."""
    SQL = """
          DROP TABLE IF EXISTS post_topic_mapping CASCADE;
          DROP TABLE IF EXISTS global_topics CASCADE;

          CREATE TABLE global_topics
          (
              topic_id     INTEGER PRIMARY KEY,
              topic_title  TEXT,
              summary_text TEXT,
              sample_posts TEXT
          );

          CREATE TABLE post_topic_mapping
          (
              post_id  TEXT REFERENCES posts (post_id) ON DELETE CASCADE,
              topic_id INTEGER REFERENCES global_topics (topic_id) ON DELETE CASCADE,
              PRIMARY KEY (post_id, topic_id)
          ); \
          
          
          """
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SQL)
                cur.execute(LIVE_SCHEMA_SQL)  # ensure live tables as well
                conn.commit()
        logger.info("Analysis tables (global_topics, post_topic_mapping, live_runs, live_outputs) created.")
    except Exception as e:
        logger.error("Failed creating analysis tables: %s", e)


# ---------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------
def run_global_analysis():
    """Public wrapper that runs synthetic analysis (pandas or spark based on config)."""
    logger.info("Starting Stage 4: Global Analysis")
    try:
        engine = utils.get_sqlalchemy_engine()
    except Exception as e:
        logger.error("Could not get SQLAlchemy engine: %s", e)
        return False
    
    if getattr(config, "USE_SPARK_ANALYSIS", False):
        return _run_global_analysis_spark(engine)
    else:
        return _run_global_analysis_pandas(engine)


# ---------------------------------------------------------------------
# If used as script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    setup_database_tables()
    run_global_analysis()
