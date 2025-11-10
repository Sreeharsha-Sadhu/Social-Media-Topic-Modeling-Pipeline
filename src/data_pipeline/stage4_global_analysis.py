"""
stage_4_global_analysis.py
───────────────────────────────────────────────
Stage 4: Global Topic Analysis (Dual-mode: Pandas or Spark)

Performs large-scale semantic clustering and summarization of
social media posts using sentence embeddings (SentenceTransformer)
and LLM summarization (FLAN-T5-Large).

Features:
    - Dual-mode execution: Pandas or Spark (USE_SPARK_ANALYSIS flag)
    - SentenceTransformer embeddings (MiniLM)
    - KMeans clustering (sklearn / Spark MLlib)
    - Abstractive LLM summarization + title generation
    - Progress bars (tqdm)
    - Topic samples + summaries persisted in PostgreSQL
"""

import re
import time
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple

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
_model_cache = {}
SAMPLE_JOINER = " ||| "  # DB-safe sample joiner


# ───────────────────────────────────────────────
# Model Loading & Caching
# ───────────────────────────────────────────────
def get_llm_pipeline():
    """Load and cache FLAN-T5-Large summarization model."""
    if "llm" not in _model_cache:
        logger.info("Loading FLAN-T5-Large (770M parameters)...")
        model_name = "google/flan-t5-large"

        if not torch.cuda.is_available():
            logger.warning("CUDA unavailable. Running on CPU (slow mode).")
            _model_cache["llm"] = pipeline("text2text-generation", model=model_name)
        else:
            _model_cache["llm"] = pipeline(
                "text2text-generation",
                model=model_name,
                device=0,
                dtype=torch.float16
            )
        logger.info("✅ FLAN-T5-Large ready.")
    return _model_cache["llm"]


# ───────────────────────────────────────────────
# Text Preprocessing
# ───────────────────────────────────────────────
def restore_readable_text(text: str) -> str:
    """Re-insert spaces and fix concatenations after normalization."""
    if not text:
        return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-z])', r'\1 \2', text)
    return re.sub(r'\s{2,}', ' ', text).strip()


def prepare_llm_text(texts: List[str]) -> str:
    """Concatenate cleaned text samples for LLM summarization."""
    cleaned = []
    for t in texts:
        if not t:
            continue
        t = t.strip()
        if not re.search(r'[.!?]$', t):
            t += "."
        cleaned.append(t.capitalize())
    return " ".join(cleaned)


# ───────────────────────────────────────────────
# LLM Summarization Helpers
# ───────────────────────────────────────────────
def summarize_topic_text(generator, full_text: str) -> str:
    """Generate a chunked abstractive summary of long text."""
    if not full_text:
        return "(No posts in this topic.)"

    chunks = [full_text[i:i + 3000] for i in range(0, len(full_text), 3000)]
    partial_summaries = []

    for chunk in chunks:
        prompt = (
            "Summarize this Reddit-style discussion into a concise paragraph "
            "capturing its theme and sentiment. Avoid repetition.\n\n"
            f"{chunk}\n\nSummary:"
        )
        try:
            part = generator(
                prompt,
                truncation=True,
                max_new_tokens=120,
                do_sample=False,
                no_repeat_ngram_size=3
            )[0]['generated_text'].strip()
            if part:
                partial_summaries.append(part)
        except Exception as e:
            logger.warning(f"Partial summary generation failed: {e}")

    if not partial_summaries:
        return "(No coherent summary generated.)"

    final_prompt = (
        "Combine the following short summaries into one cohesive paragraph "
        "describing the main ideas and tone. Avoid repetition.\n\n"
        f"{' '.join(partial_summaries)}\n\nFinal summary:"
    )
    try:
        return generator(
            final_prompt,
            truncation=True,
            max_new_tokens=180,
            min_length=40,
            do_sample=False
        )[0]['generated_text'].strip()
    except Exception:
        return " ".join(partial_summaries)[:1200]


def generate_topic_title(generator, sample_posts: List[str], summary_text: str) -> str:
    """Generate a short, catchy 3–7 word title for the topic."""
    try:
        joined_samples = " ".join(sample_posts)[:700]
        prompt = (
            "Write a concise 3-7 word title summarizing this discussion theme.\n\n"
            f"Posts: {joined_samples}\nSummary: {summary_text}\n\nTitle:"
        )
        title = generator(prompt, truncation=True, max_new_tokens=12, do_sample=False)[0]["generated_text"]
        title = re.sub(r'[^A-Za-z0-9\s-]', '', title).strip()
        return " ".join(title.split()[:7]) or "Untitled Topic"
    except Exception:
        return "Untitled Topic"


# ───────────────────────────────────────────────
# Pandas-based Analysis
# ───────────────────────────────────────────────
def _run_global_analysis_pandas(engine) -> bool:
    """Execute full topic analysis pipeline using Pandas."""
    logger.info("🔍 Starting Pandas-based topic analysis...")

    try:
        sql_query = """
            SELECT post_id, cleaned_content
            FROM posts
            WHERE cleaned_content IS NOT NULL AND cleaned_content != ''
        """
        pd_posts = pd.read_sql(sql_query, engine)

        if len(pd_posts) < 20:
            logger.warning(f"Insufficient posts ({len(pd_posts)}). Aborting.")
            return False
        logger.info(f"Loaded {len(pd_posts)} posts.")

        # Embeddings
        logger.info("Generating SentenceTransformer embeddings...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        embeddings = sbert.encode(pd_posts["cleaned_content"].tolist(), batch_size=16, show_progress_bar=True)

        # Clustering
        num_topics = getattr(config, "PANDAS_ANALYSIS_TOPICS", 20)
        logger.info(f"Clustering posts into {num_topics} topics...")
        kmeans = SkKMeans(n_clusters=num_topics, random_state=42, n_init=10)
        pd_posts["topic_id"] = kmeans.fit_predict(embeddings)

        # Summarization
        logger.info("Summarizing clustered topics with FLAN-T5-Large...")
        generator = get_llm_pipeline()
        topic_rows = []

        for topic_id, group in tqdm(pd_posts.groupby("topic_id"), total=num_topics, desc="Summarizing Topics"):
            unique_posts = list(dict.fromkeys(group["cleaned_content"].tolist()))
            restored_posts = [restore_readable_text(p) for p in unique_posts]
            full_text = prepare_llm_text(restored_posts)[:3500]
            sample_posts = [p for p in restored_posts if p][:3]

            summary = summarize_topic_text(generator, full_text)
            title = generate_topic_title(generator, sample_posts, summary)

            topic_rows.append((int(topic_id), title, summary, SAMPLE_JOINER.join(sample_posts)))

        logger.info("✅ All topic summaries generated successfully.")

        # Save to DB
        pd_summaries = pd.DataFrame(topic_rows, columns=["topic_id", "topic_title", "summary_text", "sample_posts"])
        pd_mappings = pd_posts[["post_id", "topic_id"]]

        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE global_topics, post_topic_mapping RESTART IDENTITY CASCADE;")
                conn.commit()

        pd_summaries.to_sql("global_topics", engine, if_exists="append", index=False)
        pd_mappings.to_sql("post_topic_mapping", engine, if_exists="append", index=False)
        logger.info("📁 Analysis results written to database.")
        return True

    except Exception as e:
        logger.error(f"🚨 Pandas Analysis failed: {e}")
        return False

    finally:
        if "engine" in locals():
            engine.dispose()
        logger.info("Pandas Analysis finished.")


# ───────────────────────────────────────────────
# Spark-based Analysis
# ───────────────────────────────────────────────
def _run_global_analysis_spark(engine) -> bool:
    """Execute distributed topic analysis using Spark."""
    from src.core.utils import get_spark_session

    logger.info("⚡ Running Spark-based topic analysis ...")
    spark = get_spark_session()

    try:
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
            logger.warning(f"Insufficient posts ({post_count}). Aborting.")
            return False

        # Embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        posts_list = [r.cleaned_content for r in df_posts.select("cleaned_content").collect()]
        embeddings = sbert.encode(posts_list, batch_size=32, show_progress_bar=True)
        np_emb = np.array(embeddings)

        num_topics = config.SPARK_ANALYSIS_TOPICS
        embed_df = spark.createDataFrame(
            [(i, Vectors.dense(vec)) for i, vec in enumerate(np_emb)],
            ["index", "features"]
        )

        try:
            kmeans = SparkKMeans(k=num_topics, seed=42, featuresCol="features", predictionCol="topic_id")
            model = kmeans.fit(embed_df)
            clustered = model.transform(embed_df).select("index", "topic_id")
        except Exception:
            logger.warning("Spark KMeans failed. Falling back to sklearn.")
            preds = SkKMeans(n_clusters=num_topics, random_state=42, n_init=10).fit_predict(np_emb)
            clustered = spark.createDataFrame(pd.DataFrame({"index": range(len(preds)), "topic_id": preds}))

        joined = (
            df_posts.withColumn("index", F.monotonically_increasing_id())
            .join(clustered, on="index")
            .select("post_id", "cleaned_content", "topic_id")
        )

        generator = get_llm_pipeline()
        topic_rows = []

        for topic_id in tqdm(range(num_topics), desc="Summarizing Topics"):
            topic_texts = [
                restore_readable_text(r.cleaned_content)
                for r in joined.filter(F.col("topic_id") == topic_id).select("cleaned_content").collect()
            ]
            if not topic_texts:
                continue

            unique = list(dict.fromkeys(topic_texts))
            full_text = prepare_llm_text(unique)[:4000]
            sample_posts = [p for p in unique if p][:3]
            summary = summarize_topic_text(generator, full_text)
            title = generate_topic_title(generator, sample_posts, summary)

            topic_rows.append((topic_id, title, summary, SAMPLE_JOINER.join(sample_posts)))

        pd_summaries = pd.DataFrame(topic_rows, columns=["topic_id", "topic_title", "summary_text", "sample_posts"])
        pd_mappings = joined.select("post_id", "topic_id").toPandas()

        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE global_topics, post_topic_mapping RESTART IDENTITY CASCADE;")
                conn.commit()

        pd_summaries.to_sql("global_topics", engine, if_exists="append", index=False)
        pd_mappings.to_sql("post_topic_mapping", engine, if_exists="append", index=False)
        logger.info("✅ Spark analysis completed and results saved.")
        return True

    except Exception as e:
        logger.error(f"🚨 Spark Analysis Error: {e}")
        return False

    finally:
        logger.info("Spark Analysis finished.")


# ───────────────────────────────────────────────
# Entrypoints
# ───────────────────────────────────────────────
def setup_database_tables() -> None:
    """Recreate global analysis tables."""
    SQL = """
        DROP TABLE IF EXISTS post_topic_mapping CASCADE;
        DROP TABLE IF EXISTS user_topics CASCADE;
        DROP TABLE IF EXISTS global_topics CASCADE;

        CREATE TABLE global_topics (
            topic_id     INTEGER PRIMARY KEY,
            topic_title  TEXT,
            summary_text TEXT,
            sample_posts TEXT
        );

        CREATE TABLE post_topic_mapping (
            post_id  TEXT REFERENCES posts (post_id) ON DELETE CASCADE,
            topic_id INTEGER REFERENCES global_topics (topic_id) ON DELETE CASCADE,
            PRIMARY KEY (post_id, topic_id)
        );
    """
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SQL)
                conn.commit()
        logger.info("✅ Global analysis tables created successfully.")
    except Exception as e:
        logger.error(f"🚨 Failed to create analysis tables: {e}")


def run_global_analysis() -> bool:
    """Main Stage 4 entrypoint."""
    logger.info("📊 Starting Stage 4: Global Topic Analysis")
    engine = utils.get_sqlalchemy_engine()

    if config.USE_SPARK_ANALYSIS:
        return _run_global_analysis_spark(engine)
    else:
        return _run_global_analysis_pandas(engine)


if __name__ == "__main__":
    setup_database_tables()
    run_global_analysis()
