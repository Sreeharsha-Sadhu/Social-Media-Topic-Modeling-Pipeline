"""
stage4_global_analysis.py

Stage 4: Global Topic Analysis (Dual-mode: Pandas or Spark) + Live feed integration

Enhancements in this regenerated version:
 - improved title generation (summary -> keywords -> title)
 - sentiment classification per-topic
 - extraction of 3 key bullet points per topic (LLM-assisted)
 - lightweight evaluation metrics for each summary
 - progress bars for chunk summarization, title-gen, bullets-gen, and clustering
 - backward-compatible persistence flow (global_topics + post_topic_mapping + live tables)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans as SkKMeans
from tqdm import tqdm
from transformers import pipeline

from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql import functions as F

from src.core import config, utils
from src.core.logging_config import get_logger

logger = get_logger(__name__)

_model_cache: Dict[str, Any] = {}
SAMPLE_JOINER = " ||| "

# SQL for live tables (ensure exists)
LIVE_SCHEMA_SQL = """
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


# -------------------------
# Model loaders / pipelines
# -------------------------
def get_llm_pipeline():
    """Text2text generation for summarization, title generation, bullets extraction."""
    if "llm" not in _model_cache:
        logger.info("Loading FLAN-T5-Large model (this may take a while)...")
        model_name = "google/flan-t5-large"
        if not torch.cuda.is_available():
            logger.warning("CUDA not available; LLM will run on CPU (slow).")
            _model_cache["llm"] = pipeline("text2text-generation", model=model_name)
        else:
            _model_cache["llm"] = pipeline("text2text-generation", model=model_name, device=0, dtype=torch.float16)
        logger.info("LLM pipeline ready.")
    return _model_cache["llm"]


def get_sentence_transformer():
    """SentenceTransformer model for embeddings."""
    if "sbert" not in _model_cache:
        logger.info("Loading SentenceTransformer 'all-MiniLM-L6-v2' ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_cache["sbert"] = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        logger.info("SentenceTransformer ready.")
    return _model_cache["sbert"]


def get_sentiment_pipeline():
    """Sentiment analysis pipeline (cached)."""
    if "sentiment" not in _model_cache:
        logger.info("Loading sentiment-analysis pipeline...")
        _model_cache["sentiment"] = pipeline("sentiment-analysis")
        logger.info("Sentiment pipeline ready.")
    return _model_cache["sentiment"]


# -------------------------
# Text helpers
# -------------------------
def restore_readable_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-z])', r'\1 \2', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def prepare_llm_text(texts: List[str]) -> str:
    cleaned = []
    for t in texts:
        if not t:
            continue
        t = t.strip()
        if not re.search(r'[.!?]$', t):
            t = t + "."
        cleaned.append(t.capitalize())
    return " ".join(cleaned)


# -------------------------
# Summarization helpers
# -------------------------
def summarize_topic_text(generator, full_text: str, chunk_size: int = 2800) -> str:
    """
    Abstractive summarization using chunking + meta-summary.
    Returns a single cohesive paragraph (no meta-statements).
    """
    if not full_text:
        return "(No meaningful content available.)"

    # split into chunks
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    partial_summaries = []

    for chunk in tqdm(chunks, desc="Chunk summaries", unit="chunk"):
        prompt = (
            "Write a clear, concise paragraph summarizing the ideas, arguments, and topics in the text below. "
            "Do NOT mention that these are posts or say 'these posts'. Focus on content, not on the fact they are posts.\n\n"
            f"{chunk}\n\nSummary:"
        )
        try:
            out = generator(prompt, truncation=True, max_new_tokens=160, do_sample=False)
            text = out[0].get("generated_text") or out[0].get("text") or ""
            if text:
                partial_summaries.append(text.strip())
        except Exception as e:
            logger.warning("Chunk summarization failed: %s", e)

    if not partial_summaries:
        return "(Unable to generate summary.)"

    joined = " ".join(partial_summaries)
    meta_prompt = (
        "Combine and condense the following short summaries into a single polished paragraph. "
        "Remove repetition, keep key themes, and present a natural flowing paragraph (no meta commentary).\n\n"
        f"{joined}\n\nFinal summary:"
    )
    try:
        out = generator(meta_prompt, truncation=True, max_new_tokens=220, do_sample=False)
        final = out[0].get("generated_text") or out[0].get("text") or ""
        return final.strip() if final else " ".join(partial_summaries)[:1200]
    except Exception:
        return " ".join(partial_summaries)[:1200]


def generate_topic_title_from_summary(generator, summary_text: str, max_words: int = 7) -> str:
    """
    Two-step title generation:
      1) extract keywords from summary_text (simple freq-based)
      2) ask the LLM to craft a 3-7 word title from those keywords
    This produces more focused, meaningful titles.
    """
    # Step A: lightweight keyword extraction (top N non-stopwords)
    keywords = extract_keywords(summary_text, top_k=8)
    kw_str = ", ".join(keywords) if keywords else summary_text[:200]

    title_prompt = (
        "Using the keywords listed, write a short, catchy 3-7 word title that summarizes the main theme. "
        "Do NOT use words like 'posts' or 'Reddit'. Only output the title.\n\n"
        f"Keywords: {kw_str}\n\nTitle:"
    )
    try:
        out = generator(title_prompt, truncation=True, max_new_tokens=12, do_sample=False)
        t = out[0].get("generated_text") or out[0].get("text") or ""
        t = re.sub(r'[^A-Za-z0-9\s-]', '', t).strip()
        words = [w for w in t.split() if w]
        if not words:
            # fallback to top keywords joined
            return " ".join(keywords[:max_words]) or "Untitled Topic"
        return " ".join(words[:max_words])
    except Exception as e:
        logger.warning("Title generation failed: %s", e)
        return " ".join(keywords[:max_words]) or "Untitled Topic"


def extract_bullets(generator, summary_text: str, num_bullets: int = 3) -> List[str]:
    """
    Use LLM to extract 3 concise bullet points from the summary text.
    Returns list of short bullets.
    """
    prompt = (
        "From the paragraph below, extract the top {} concise bullet points (each 8-20 words). "
        "Write each bullet on its own line and avoid meta-comments.\n\n"
        f"{summary_text}\n\nBullets:".format(num_bullets)
    )
    bullets = []
    try:
        out = generator(prompt, truncation=True, max_new_tokens=140, do_sample=False)
        text = out[0].get("generated_text") or out[0].get("text") or ""
        # split by newlines and clean
        for line in text.splitlines():
            line = line.strip("-• \t")
            if line:
                bullets.append(line.strip())
        # ensure we have at most num_bullets
        return bullets[:num_bullets] or []
    except Exception as e:
        logger.warning("Bullet extraction failed: %s", e)
        return []


def extract_keywords(text: str, top_k: int = 8) -> List[str]:
    """
    Lightweight keyword extraction: normalize, remove small stopwords, then return top_k frequent tokens.
    This is intentionally simple and fast (no external deps).
    """
    if not text:
        return []
    # basic stoplist
    stopwords = {
        "the", "and", "a", "an", "to", "of", "for", "in", "on", "is", "are", "was", "were",
        "that", "this", "with", "it", "as", "by", "be", "from", "or", "at", "its", "has",
        "have", "will", "but", "not", "they", "their"
    }
    tokens = re.findall(r"\b[a-zA-Z0-9\-']+\b", text.lower())
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    if not tokens:
        return []
    counts = Counter(tokens)
    common = [w for w, _ in counts.most_common(top_k)]
    return common


# -------------------------
# Evaluation metrics
# -------------------------
def evaluate_summary(original_texts: List[str], summary_text: str) -> Dict[str, Any]:
    """
    Lightweight evaluation:
      - compression_ratio: summary_words / original_words
      - lexical_overlap: overlap unique words / unique original words
      - summary_length: number of words in summary
    These are simple proxies (not ROUGE) but help detect extremes.
    """
    orig_joined = " ".join(original_texts or [])
    orig_tokens = re.findall(r"\b[a-zA-Z0-9']+\b", orig_joined.lower())
    sum_tokens = re.findall(r"\b[a-zA-Z0-9']+\b", (summary_text or "").lower())

    orig_unique = set(orig_tokens)
    sum_unique = set(sum_tokens)

    orig_count = max(1, len(orig_tokens))
    sum_count = max(1, len(sum_tokens))

    compression_ratio = round(sum_count / orig_count, 4)
    overlap = round((len(sum_unique & orig_unique) / (len(orig_unique) or 1)), 4)
    metrics = {
        "compression_ratio": compression_ratio,
        "lexical_overlap": overlap,
        "summary_length": sum_count
    }
    return metrics


# -------------------------
# Core topic-model runner (in-memory)
# -------------------------
def run_topic_model_on_text_list(
        texts: List[str],
        num_topics: int = 6,
        full_pipeline_threshold: int = 50,
) -> Dict[str, Any]:
    """
    Run topic modelling/summarization on a list of texts.
    Returns a dictionary with topics, mappings, and metadata. Each topic includes:
       - topic_index (int)
       - title (str)
       - summary (str)
       - sample_posts (List[str])
       - bullets (List[str])
       - sentiment (dict label/score)
       - keywords (List[str])
       - metrics (dict)
       - post_indices (List[int])
    """
    texts = [t for t in (texts or []) if t and t.strip()]
    if not texts:
        return {"type": "empty", "topics": [], "mappings": [], "raw_texts": []}

    restored = [restore_readable_text(t) for t in texts]
    generator = get_llm_pipeline()
    n = len(restored)
    result: Dict[str, Any] = {"raw_texts": restored, "topics": []}

    # Small workload -> single LLM summary
    if n <= full_pipeline_threshold:
        logger.info("Using lightweight single-LLM summarization for %d posts.", n)
        blob = prepare_llm_text(restored)[:4000]
        # chunk+meta summarization to keep structure consistent (use progress bars for stages)
        summary = summarize_topic_text(generator, blob)
        title = generate_topic_title_from_summary(generator, summary)
        bullets = extract_bullets(generator, summary)
        sentiment = get_sentiment_pipeline()(summary[:512])[0] if summary else {"label": "NEUTRAL", "score": 0.0}
        metrics = evaluate_summary(restored, summary)

        topic = {
            "topic_index": 0,
            "title": title,
            "summary": summary,
            "sample_posts": restored[:3],
            "bullets": bullets,
            "sentiment": sentiment,
            "keywords": extract_keywords(summary, top_k=6),
            "metrics": metrics,
            "post_indices": list(range(n))
        }
        mappings = [{"post_index": i, "topic_index": 0} for i in range(n)]
        result.update({"type": "single_summary", "topics": [topic], "mappings": mappings})
        return result

    # Full pipeline: embeddings + KMeans
    logger.info("Running full pipeline for %d posts: embeddings + KMeans + summarization", n)
    sbert = get_sentence_transformer()
    embeddings = sbert.encode(restored, batch_size=32, show_progress_bar=True)
    np_emb = np.array(embeddings)

    num_topics = max(2, min(num_topics, n))
    try:
        km = SkKMeans(n_clusters=num_topics, random_state=42, n_init=10)
        preds = km.fit_predict(np_emb)
    except Exception as e:
        logger.warning("Sklearn KMeans failed: %s", e)
        preds = np.zeros(n, dtype=int)

    topics: List[Dict[str, Any]] = []
    # iterate clusters with progress
    cluster_range = range(int(num_topics))
    for tid in tqdm(cluster_range, desc="Generating topic summaries", unit="topic"):
        indices = [i for i, p in enumerate(preds) if int(p) == tid]
        if not indices:
            continue
        posts_for_topic = [restored[i] for i in indices]
        unique_posts = list(dict.fromkeys(posts_for_topic))
        full_text = prepare_llm_text(unique_posts)[:3500]

        # summarization + title + bullets + sentiment + metrics
        try:
            summary = summarize_topic_text(generator, full_text)
        except Exception as e:
            logger.warning("Summary generation failed for topic %d: %s", tid, e)
            summary = "Error generating summary."

        # title via keywords
        title = generate_topic_title_from_summary(generator, summary)
        bullets = extract_bullets(generator, summary)
        sentiment = get_sentiment_pipeline()(summary[:512])[0] if summary else {"label": "NEUTRAL", "score": 0.0}
        metrics = evaluate_summary(unique_posts, summary)
        keywords = extract_keywords(summary, top_k=6)

        topics.append({
            "topic_index": int(tid),
            "title": title,
            "summary": summary,
            "sample_posts": unique_posts[:3],
            "bullets": bullets,
            "sentiment": sentiment,
            "keywords": keywords,
            "metrics": metrics,
            "post_indices": indices
        })

    mappings = [{"post_index": i, "topic_index": int(preds[i])} for i in range(len(preds))]
    result.update({"type": "clustered", "topics": topics, "mappings": mappings})
    return result


# -------------------------
# DB helpers & persistence
# -------------------------
def _ensure_live_tables():
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(LIVE_SCHEMA_SQL)
                conn.commit()
        logger.debug("Ensured live tables exist.")
    except Exception as e:
        logger.warning("Could not ensure live tables: %s", e)


def _save_live_output(user_id: str, source: str, payload: Dict[str, Any]) -> int:
    """
    Always serialize payload to JSON before inserting.
    Prevents dict objects from being mistakenly stored.
    """
    try:
        payload_json = json.dumps(payload)  # ALWAYS serialize!
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO live_outputs (user_id, source, payload_json)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (user_id, source, payload_json))
                out_id = cur.fetchone()[0]
                conn.commit()
        return int(out_id)
    except Exception as e:
        logger.error("Failed to save live output: %s", e)
        raise




def _upsert_live_run(user_id: str, source: str, post_count: int, output_id: int):
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
    """
    Loads payload_json safely:
      - If the row contains a dict, return it directly.
      - If it's a string, parse it as JSON.
      - If parsing fails, safely return None.
    """
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT payload_json FROM live_outputs
                    WHERE id = %s
                """, (output_id,))
                row = cur.fetchone()
    except Exception as e:
        logger.warning("Failed to load live output %s: %s", output_id, e)
        return None

    if not row:
        return None

    payload_raw = row[0]

    # CASE 1: already a dict
    if isinstance(payload_raw, dict):
        return payload_raw

    # CASE 2: JSON string → parse
    if isinstance(payload_raw, str):
        try:
            return json.loads(payload_raw)
        except Exception as e:
            logger.error("Corrupt JSON in live_outputs.id=%s: %s", output_id, e)
            return None

    # CASE 3: some other type — return safe fallback
    logger.error("Unexpected payload_json type for live_outputs.id=%s: %r", output_id, type(payload_raw))
    return None



def _persist_topics_and_mappings(user_id: str, source: str, topics: List[Dict[str, Any]], post_ids: List[str]) -> None:
    """
    Insert topics into global_topics (letting DB assign topic_id) and insert mappings into post_topic_mapping.
    Also insert into live_topics/live_post_topic_mapping for traceability.
    """
    if not topics:
        return

    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                # insert each topic into global_topics and live_topics, map internal topic_index -> db topic_id and live_topic_id
                topic_index_to_global_dbid = {}
                topic_index_to_live_dbid = {}

                for t in topics:
                    # insert into global_topics (topic_id is serial)
                    cur.execute(
                        "INSERT INTO global_topics (topic_title, summary_text, sample_posts) VALUES (%s, %s, %s) RETURNING topic_id",
                        (t.get("title"), t.get("summary"), SAMPLE_JOINER.join(t.get("sample_posts", []) or []))
                    )
                    global_dbid = cur.fetchone()[0]
                    topic_index_to_global_dbid[int(t["topic_index"])] = global_dbid

                    # also save into live_topics for per-run traceability (user_id/source optional)
                    cur.execute(
                        "INSERT INTO live_topics (user_id, source, topic_title, summary_text, sample_posts) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                        (user_id or "", source or "", t.get("title"), t.get("summary"), SAMPLE_JOINER.join(t.get("sample_posts", []) or []))
                    )
                    live_dbid = cur.fetchone()[0]
                    topic_index_to_live_dbid[int(t["topic_index"])] = live_dbid

                # insert mappings into post_topic_mapping and live_post_topic_mapping
                mapping_pairs = []
                live_mapping_pairs = []
                for t in topics:
                    tidx = int(t["topic_index"])
                    g_dbid = topic_index_to_global_dbid.get(tidx)
                    l_dbid = topic_index_to_live_dbid.get(tidx)
                    for post_i in t.get("post_indices", []):
                        pid = post_ids[int(post_i)]
                        mapping_pairs.append((pid, g_dbid))
                        live_mapping_pairs.append((pid, l_dbid, user_id or "", source or ""))

                if mapping_pairs:
                    cur.executemany(
                        "INSERT INTO post_topic_mapping (post_id, topic_id) VALUES (%s, %s)",
                        mapping_pairs
                    )
                if live_mapping_pairs:
                    cur.executemany(
                        "INSERT INTO live_post_topic_mapping (post_ref, topic_ref, user_id, source) VALUES (%s, %s, %s, %s)",
                        live_mapping_pairs
                    )
                conn.commit()
    except Exception as e:
        logger.warning("Failed persisting topics/mappings: %s", e)


# -------------------------
# Live-run orchestration
# -------------------------
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
    fetcher_callable(user_id, since:Optional[datetime], limit:int) -> list[posts]
    Each post dict must have at least: {'post_id': str, 'text' or 'content': str, 'created_at': iso str}
    """
    logger.info("Starting live analysis for %s/%s (top_n=%d)", user_id, source, top_n)
    _ensure_live_tables()

    if not callable(fetcher_callable):
        raise ValueError("fetcher_callable must be provided and callable(user_id, since, limit)")

    last = get_last_run_for_user(user_id, source)
    since = last["last_run_at"] if (last and prefer_since_last_run and last.get("last_run_at")) else None

    try:
        posts = fetcher_callable(user_id, since=since, limit=top_n)
        if not posts:
            logger.warning("No posts returned by fetcher for %s/%s. Returning cached output if available.", user_id, source)
            if last and last.get("output_id"):
                cached = _load_live_output_by_id(last["output_id"])
                if cached:
                    return {"status": "cached_fallback", "payload": cached}
                else:
                    logger.warning("Cached output was unreadable. Returning no_data.")
                    return {"status": "no_data", "payload": None}
            return {"status": "no_data", "payload": None}
        
        # sort by created_at descending, defensive about types
        posts_sorted = sorted(posts, key=lambda p: p.get("created_at", ""), reverse=True)
        selected = posts_sorted[:top_n]

        texts = [p.get("text") or p.get("content") or "" for p in selected]
        post_ids = [p.get("post_id") or p.get("id") or str(i) for i, p in enumerate(selected)]
        processed_count = len(texts)
        logger.info("Fetched %d posts (processing %d) for %s/%s", len(posts), processed_count, user_id, source)

        if processed_count <= full_pipeline_threshold:
            logger.info("Using single-LLM summarization pipeline (<= %d posts).", full_pipeline_threshold)
            res = run_topic_model_on_text_list(texts, num_topics=1, full_pipeline_threshold=full_pipeline_threshold)
        else:
            logger.info("Using full clustered pipeline (> %d posts).", full_pipeline_threshold)
            res = run_topic_model_on_text_list(texts, num_topics=num_topics, full_pipeline_threshold=full_pipeline_threshold)

        payload = {
            "user_id": user_id,
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "post_count": processed_count,
            "result": res,
            "post_ids": post_ids
        }

        output_id = _save_live_output(user_id, source, payload)
        _upsert_live_run(user_id, source, processed_count, output_id)

        # Persist topics/mappings safely (DB-assigned ids)
        try:
            if res.get("type") in ("clustered", "single_summary"):
                # normalize topics into expected format
                topics_for_persist = []
                if res["type"] == "single_summary":
                    t = res["topics"][0]
                    topics_for_persist.append({
                        "topic_index": 0,
                        "title": t.get("title"),
                        "summary": t.get("summary"),
                        "sample_posts": t.get("sample_posts", []),
                        "post_indices": t.get("post_indices", [])
                    })
                else:
                    for t in res["topics"]:
                        topics_for_persist.append({
                            "topic_index": t.get("topic_index"),
                            "title": t.get("title"),
                            "summary": t.get("summary"),
                            "sample_posts": t.get("sample_posts", []),
                            "post_indices": t.get("post_indices", [])
                        })
                _persist_topics_and_mappings(user_id, source, topics_for_persist, post_ids)
        except Exception as e:
            logger.warning("Failed to persist topics/mappings from live run: %s", e)

        return {"status": "ok", "payload": payload}

    except Exception as e:
        logger.exception("Live analysis failed for %s/%s: %s", user_id, source, e)
        last = get_last_run_for_user(user_id, source)
        if last and last.get("output_id"):
            cached = _load_live_output_by_id(last["output_id"])
            return {"status": "error_with_cache", "error": str(e), "payload": cached}
        return {"status": "error", "error": str(e), "payload": None}


# -------------------------
# Batch analysis (pandas / spark) (keeps behavior + persists using the improved persister)
# -------------------------
def _run_global_analysis_pandas(engine) -> bool:
    try:
        sql_query = """
            SELECT post_id, cleaned_content
            FROM posts
            WHERE cleaned_content IS NOT NULL AND cleaned_content != ''
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
        pd_posts["topic_index"] = km.fit_predict(np.array(embeddings))

        generator = get_llm_pipeline()
        topics_for_persist = []

        for topic_idx, group in tqdm(pd_posts.groupby("topic_index"), total=num_topics, desc="Summarizing Topics"):
            unique_posts = list(dict.fromkeys(group["cleaned_content"].tolist()))
            restored = [restore_readable_text(p) for p in unique_posts]
            full_text = prepare_llm_text(restored)[:3500]
            summary = summarize_topic_text(generator, full_text)
            title = generate_topic_title_from_summary(generator, summary)
            bullets = extract_bullets(generator, summary)
            sentiment = get_sentiment_pipeline()(summary[:512])[0] if summary else {"label": "NEUTRAL", "score": 0.0}
            metrics = evaluate_summary(restored, summary)

            topics_for_persist.append({
                "topic_index": int(topic_idx),
                "title": title,
                "summary": summary,
                "sample_posts": restored[:3],
                "bullets": bullets,
                "sentiment": sentiment,
                "keywords": extract_keywords(summary, top_k=6),
                "metrics": metrics,
                "post_indices": list(group.index)
            })

        # Persist: truncate tables then insert via improved persister
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE global_topics, post_topic_mapping RESTART IDENTITY CASCADE;")
                conn.commit()

        # build post_ids list in the same order as pd_posts (index aligns)
        post_ids = pd_posts["post_id"].tolist()
        _persist_topics_and_mappings("", "batch", topics_for_persist, post_ids)
        logger.info("Synthetic analysis complete and saved to DB.")
        return True
    except Exception as e:
        logger.exception("Pandas analysis failed: %s", e)
        return False


def _run_global_analysis_spark(engine) -> bool:
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
                                         ["index", "features"]).repartition(max(2, spark.sparkContext.defaultParallelism)).cache()
        try:
            kmeans = SparkKMeans(k=num_topics, seed=42, featuresCol="features", predictionCol="prediction", maxIter=20)
            model = kmeans.fit(embed_df)
            clustered = model.transform(embed_df).select("index", F.col("prediction").alias("topic_index"))
        except Exception:
            logger.warning("Spark KMeans failed; falling back to sklearn on driver.")
            np_emb2 = np.array([r.features.toArray() for r in embed_df.collect()])
            preds = SkKMeans(n_clusters=num_topics, random_state=42, n_init=10).fit_predict(np_emb2)
            clustered = spark.createDataFrame(pd.DataFrame({"index": range(len(preds)), "topic_index": preds}))

        posts_with_index = df_posts.withColumn("index", F.monotonically_increasing_id())
        joined_df = posts_with_index.join(clustered, on="index").select("post_id", "cleaned_content", "topic_index")
        generator = get_llm_pipeline()

        topics_for_persist = []
        pandas_joined = joined_df.toPandas()

        for topic_idx in range(num_topics):
            group = pandas_joined[pandas_joined["topic_index"] == topic_idx]
            if group.empty:
                continue
            unique_posts = list(dict.fromkeys(group["cleaned_content"].tolist()))
            restored = [restore_readable_text(p) for p in unique_posts]
            full_text = prepare_llm_text(restored)[:4000]
            summary = summarize_topic_text(generator, full_text)
            title = generate_topic_title_from_summary(generator, summary)
            bullets = extract_bullets(generator, summary)
            sentiment = get_sentiment_pipeline()(summary[:512])[0] if summary else {"label": "NEUTRAL", "score": 0.0}
            metrics = evaluate_summary(restored, summary)

            topics_for_persist.append({
                "topic_index": int(topic_idx),
                "title": title,
                "summary": summary,
                "sample_posts": restored[:3],
                "bullets": bullets,
                "sentiment": sentiment,
                "keywords": extract_keywords(summary, top_k=6),
                "metrics": metrics,
                "post_indices": group.index.tolist()
            })

        # persist: truncate and insert
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE global_topics, post_topic_mapping RESTART IDENTITY CASCADE;")
                conn.commit()

        post_ids = pandas_joined["post_id"].tolist()
        _persist_topics_and_mappings("", "batch", topics_for_persist, post_ids)
        logger.info("Spark analysis complete.")
        return True
    except Exception as e:
        logger.exception("Spark analysis error: %s", e)
        return False


# -------------------------
# DB setup: global + live tables
# -------------------------
def setup_database_tables():
    SQL = """
          DROP TABLE IF EXISTS post_topic_mapping CASCADE;
          DROP TABLE IF EXISTS global_topics CASCADE;

          CREATE TABLE global_topics
          (
              topic_id     SERIAL PRIMARY KEY,
              topic_title  TEXT,
              summary_text TEXT,
              sample_posts TEXT
          );

          CREATE TABLE post_topic_mapping
          (
              post_id  TEXT REFERENCES posts (post_id) ON DELETE CASCADE,
              topic_id INTEGER REFERENCES global_topics (topic_id) ON DELETE CASCADE,
              PRIMARY KEY (post_id, topic_id)
          );
          """
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SQL)
                cur.execute(LIVE_SCHEMA_SQL)
                conn.commit()
        logger.info("Analysis tables created/ensured.")
    except Exception as e:
        logger.exception("Failed creating analysis tables: %s", e)



# -------------------------
# Public wrapper
# -------------------------
def run_global_analysis() -> bool:
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


if __name__ == "__main__":
    setup_database_tables()
    run_global_analysis()
