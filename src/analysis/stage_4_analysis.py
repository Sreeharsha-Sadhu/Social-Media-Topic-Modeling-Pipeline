# stage_4_analysis.py
# ---------------------------------------------------------------------
# Stage 4: Global Topic Analysis (Dual-mode: Pandas or Spark)
# Features:
# - SentenceTransformer embeddings
# - KMeans clustering (Spark MLlib or sklearn fallback)
# - LLM summarization with progress bar
# - Topic title + sample posts (saved to DB)
# ---------------------------------------------------------------------

import re
import time
from typing import List

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

from src.common import utils
from src.common.utils import get_spark_session
from src.config import settings

# ---------------------------------------------------------------------
# Global cache for model instances
# ---------------------------------------------------------------------
_model_cache = {}

SAMPLE_JOINER = " ||| "  # used to join sample posts for DB-friendly storage


def get_llm_pipeline():
    """Load (and cache) the FLAN-T5-Large model for text summarization/title."""
    if "llm" not in _model_cache:
        print("\n--- Loading FLAN-T5-Large model (770M params) ---")
        model_name = "google/flan-t5-large"

        if not torch.cuda.is_available():
            print("⚠️  CUDA not available. Using CPU — this will be very slow.")
            _model_cache["llm"] = pipeline("text2text-generation", model=model_name)
        else:
            _model_cache["llm"] = pipeline(
                "text2text-generation",
                model=model_name,
                device=0,
                dtype=torch.float16
            )
    return _model_cache["llm"]


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def restore_readable_text(text: str) -> str:
    """Try to restore spacing and readability after light normalization."""
    if not text:
        return ""
    # Re-insert spaces before uppercase and between letters/digits that were concatenated
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-z])', r'\1 \2', text)
    # collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def prepare_llm_text(texts: List[str]) -> str:
    """Prepare a combined text blob for the LLM, preserving sentence structure."""
    # capitalize sentences for readability by LLM
    cleaned = []
    for t in texts:
        if not t:
            continue
        t = t.strip()
        # ensure ends with punctuation where possible
        if not re.search(r'[.!?]$', t):
            t = t + "."
        cleaned.append(t.capitalize())
    return " ".join(cleaned)


def summarize_topic_text(generator, full_text: str) -> str:
    """
    Generate a multi-stage summary for possibly long topic text.
    - Breaks into chunks, summarizes each, then meta-summarizes.
    """
    if not full_text:
        return "(No posts in this topic.)"

    # chunk size tuned to ~3k chars to keep prompt sizes reasonable
    chunk_size = 3000
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    partial_summaries = []

    for i, chunk in enumerate(chunks):
        prompt = (
            "Summarize this short Reddit-style discussion into a concise paragraph "
            "that captures the theme and sentiment. Keep it readable and avoid repetition.\n\n"
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
        except Exception:
            # conservative fallback
            part = ""
        if part:
            partial_summaries.append(part)

    # combine partials
    if not partial_summaries:
        return "(No coherent summary generated.)"

    joined = " ".join(partial_summaries)
    final_prompt = (
        "Combine the following mini-summaries into a single cohesive paragraph "
        "describing the main discussion themes and sentiment. Avoid repetition.\n\n"
        f"{joined}\n\nFinal summary:"
    )
    try:
        final_summary = generator(
            final_prompt,
            truncation=True,
            max_new_tokens=180,
            min_length=40,
            do_sample=False,
            no_repeat_ngram_size=3
        )[0]['generated_text'].strip()
    except Exception:
        final_summary = " ".join(partial_summaries)[:1200]

    return final_summary or "(No coherent summary generated.)"


def generate_topic_title(generator, sample_posts: List[str], summary_text: str) -> str:
    """
    Generate a short (3-7 word) title that captures the theme.
    Best-effort; returns 'Untitled Topic' on failure.
    """
    try:
        joined_samples = " ".join(sample_posts)[:700]
        prompt = (
            "Create a concise, catchy 3-7 word title (no punctuation) summarizing the theme "
            "of these social media posts and their summary.\n\n"
            f"Posts: {joined_samples}\n\nSummary: {summary_text}\n\nTitle:"
        )
        title = generator(
            prompt,
            truncation=True,
            max_new_tokens=12,
            do_sample=False,
            no_repeat_ngram_size=2
        )[0]["generated_text"].strip()
        # clean up any trailing punctuation and excessive whitespace
        title = re.sub(r'[^A-Za-z0-9\s-]', '', title).strip()
        if len(title.split()) > 10:
            # enforce shortness if model returns long phrase
            title = " ".join(title.split()[:7])
        return title or "Untitled Topic"
    except Exception:
        return "Untitled Topic"


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------
def run_global_analysis():
    print("\n--- Starting Stage 4: Global Analysis ---")
    mode = "Spark" if getattr(settings, "USE_SPARK_ANALYSIS", False) else "Pandas"
    print(f"Analysis Mode: {mode}\n")

    try:
        engine = utils.get_sqlalchemy_engine()
    except Exception as e:
        print(f"🚨 Could not create SQLAlchemy engine: {e}")
        return False

    if getattr(settings, "USE_SPARK_ANALYSIS", False):
        return _run_global_analysis_spark(engine)
    else:
        return _run_global_analysis_pandas(engine)


# ---------------------------------------------------------------------
# Pandas-based Analysis Pipeline
# ---------------------------------------------------------------------
def _run_global_analysis_pandas(engine):
    try:
        print("Loading posts from PostgreSQL into Pandas...")
        sql_query = """
                    SELECT post_id, cleaned_content
                    FROM posts
                    WHERE cleaned_content IS NOT NULL
                      AND cleaned_content != ''
                    """
        pd_posts = pd.read_sql(sql_query, engine)

        if len(pd_posts) < 20:
            print(f"Not enough posts ({len(pd_posts)}) to analyze. Aborting.")
            return False
        print(f"Loaded {len(pd_posts)} posts.\n")

        print("Loading SentenceTransformer model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        print("Generating embeddings...")
        embeddings = sbert_model.encode(
            pd_posts['cleaned_content'].tolist(),
            batch_size=16,
            show_progress_bar=True
        )

        NUM_TOPICS = getattr(settings, "PANDAS_ANALYSIS_TOPICS", 20)
        print(f"\nClustering {len(pd_posts)} posts into {NUM_TOPICS} topics...")
        kmeans = SkKMeans(n_clusters=NUM_TOPICS, random_state=0, n_init=10)
        pd_posts['topic_id'] = kmeans.fit_predict(embeddings)

        print("\nSummarizing topics using FLAN-T5-Large (with tqdm progress bar)...")
        generator = get_llm_pipeline()
        topic_rows = []  # list of tuples: (topic_id, topic_title, summary_text, sample_posts_joined)

        for topic_id, group_df in tqdm(
                pd_posts.groupby('topic_id'),
                total=NUM_TOPICS,
                desc="Generating summaries",
                unit="topic"
        ):
            start_time = time.time()
            # deduplicate posts and restore readable text
            unique_posts = list(dict.fromkeys(group_df["cleaned_content"].tolist()))
            restored_posts = [restore_readable_text(p) for p in unique_posts]
            # prepare text blob for LLM
            full_text = prepare_llm_text(restored_posts)[:3500]

            # pick up to 3 sample posts (already 'restored')
            sample_posts = [p for p in restored_posts if p][:3]

            try:
                summary = summarize_topic_text(generator, full_text)
                # safety trim
                summary = summary.strip()
                title = generate_topic_title(generator, sample_posts, summary)
                tqdm.write(f"Topic {topic_id:02d} ✅ {title} ({time.time() - start_time:.1f}s)")
            except Exception as e:
                tqdm.write(f"⚠️ Error summarizing topic {topic_id}: {e}")
                summary = f"Error generating summary: {e}"
                title = "Untitled Topic"

            # join sample posts safely for DB storage
            sample_joined = SAMPLE_JOINER.join(sample_posts)
            topic_rows.append((int(topic_id), title, summary, sample_joined))

        print("\n✅ All topic summaries generated successfully!")

        # Prepare results DataFrames
        pd_summaries = pd.DataFrame(topic_rows, columns=["topic_id", "topic_title", "summary_text", "sample_posts"])
        pd_mappings = pd_posts[['post_id', 'topic_id']]

        print("Saving results to PostgreSQL...")
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE global_topics, post_topic_mapping RESTART IDENTITY CASCADE;")
                conn.commit()

        # to_sql: sample_posts stored as text joined with SAMPLE_JOINER
        pd_summaries.to_sql("global_topics", engine, if_exists='append', index=False)
        pd_mappings.to_sql("post_topic_mapping", engine, if_exists='append', index=False)

        print("\n✅ Analysis complete. Results saved to database.")
        print("\n--- Final Topic Summaries ---")
        print(pd_summaries.sort_values(by='topic_id').to_string(index=False))
        return True

    except Exception as e:
        print(f"\n🚨 ERROR during Pandas analysis:\n{e}")
        return False

    finally:
        try:
            if 'engine' in locals():
                engine.dispose()
        except Exception:
            pass
        print("Analysis process finished.\n")


# ---------------------------------------------------------------------
# Spark-based Analysis Pipeline
# ---------------------------------------------------------------------
def _run_global_analysis_spark(engine):
    print("--- 🚀 Running Spark Analysis ---")
    spark = get_spark_session()

    try:
        # Step 1: Load data
        print("Loading posts from PostgreSQL into Spark...")
        df_posts = spark.read \
            .format("jdbc") \
            .option("url", f"jdbc:postgresql://{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}") \
            .option("dbtable", "posts") \
            .option("user", settings.DB_USER) \
            .option("password", settings.DB_PASS) \
            .load() \
            .filter(F.col("cleaned_content").isNotNull() & (F.col("cleaned_content") != ""))

        post_count = df_posts.count()
        if post_count < 20:
            print(f"Not enough posts ({post_count}) to analyze. Aborting.")
            return False
        print(f"Loaded {post_count} posts into Spark DataFrame.\n")

        # Step 2: Generate embeddings
        print("Generating embeddings using SentenceTransformer...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        posts_list = [row.cleaned_content for row in df_posts.select("cleaned_content").collect()]
        embeddings = sbert_model.encode(posts_list, batch_size=32, show_progress_bar=True)
        np_embeddings = np.array(embeddings)

        # Step 3: Cluster embeddings
        num_topics = getattr(settings, "SPARK_ANALYSIS_TOPICS", 20)
        print(f"\nClustering embeddings (target topics={num_topics})...")
        embed_df = spark.createDataFrame(
            [(int(i), Vectors.dense(vec)) for i, vec in enumerate(np_embeddings)],
            ["index", "features"]
        ).repartition(max(2, spark.sparkContext.defaultParallelism)).cache()

        try:
            kmeans = SparkKMeans(k=num_topics, seed=42,
                                 featuresCol="features", predictionCol="prediction", maxIter=20)
            model = kmeans.fit(embed_df)
            clustered = model.transform(embed_df).select("index", F.col("prediction").alias("topic_id"))
        except Exception as e:
            print("⚠️ Spark KMeans failed. Falling back to sklearn on driver.")
            np_emb = np.array([r.features.toArray() for r in embed_df.collect()])
            sk = SkKMeans(n_clusters=num_topics, random_state=42, n_init=10)
            preds = sk.fit_predict(np_emb)
            clustered = spark.createDataFrame(pd.DataFrame({"index": range(len(preds)), "topic_id": preds}))

        # Step 4: Merge with posts
        posts_with_index = df_posts.withColumn("index", F.monotonically_increasing_id())
        joined_df = posts_with_index.join(clustered, on="index").select("post_id", "cleaned_content", "topic_id")

        # Step 5: Summarize topics
        print("\n🧠 Generating topic summaries (LLM)...")
        generator = get_llm_pipeline()
        topic_rows = []

        for topic_id in tqdm(range(num_topics), desc="Generating summaries", unit="topic"):
            # collect texts for topic
            topic_texts = [
                restore_readable_text(r.cleaned_content)
                for r in joined_df.filter(F.col("topic_id") == int(topic_id)).select("cleaned_content").collect()
            ]
            if not topic_texts:
                continue

            start_time = time.time()
            unique_posts = list(dict.fromkeys(topic_texts))
            full_text = prepare_llm_text(unique_posts)[:4000]
            sample_posts = [p for p in unique_posts if p][:3]

            try:
                summary = summarize_topic_text(generator, full_text)
                title = generate_topic_title(generator, sample_posts, summary)
                tqdm.write(f"Topic {topic_id:02d} ✅ {title} ({time.time() - start_time:.1f}s)")
            except Exception as e:
                tqdm.write(f"⚠️ Error summarizing topic {topic_id}: {e}")
                summary = f"Error generating summary: {e}"
                title = "Untitled Topic"

            sample_joined = SAMPLE_JOINER.join(sample_posts)
            topic_rows.append((int(topic_id), title, summary, sample_joined))

        print("\n✅ All summaries generated successfully!")

        # Step 6: Save results
        pd_summaries = pd.DataFrame(topic_rows, columns=["topic_id", "topic_title", "summary_text", "sample_posts"])
        pd_mappings = joined_df.select("post_id", "topic_id").toPandas()

        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE global_topics, post_topic_mapping RESTART IDENTITY CASCADE;")
                conn.commit()

        pd_summaries.to_sql("global_topics", engine, if_exists='append', index=False)
        pd_mappings.to_sql("post_topic_mapping", engine, if_exists='append', index=False)

        print("\n✅ Spark Analysis Complete!")
        return True

    except Exception as e:
        print(f"\n🚨 Spark Analysis Error: {e}")
        return False

    finally:
        print("Spark Analysis finished.\n")


# ---------------------------------------------------------------------
# Utility for table setup (first-time run)
# ---------------------------------------------------------------------
def setup_database_tables():
    print("Setting up database tables for global analysis (destructive)...")
    SQL = """
          DROP TABLE IF EXISTS post_topic_mapping CASCADE;
          DROP TABLE IF EXISTS user_topics CASCADE;
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
          );
          """

    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SQL)
                conn.commit()
        print("✅ Database tables created successfully.")
    except Exception as e:
        print(f"🚨 ERROR setting up database tables:\n{e}")


# ---------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    setup_database_tables()
    run_global_analysis()
