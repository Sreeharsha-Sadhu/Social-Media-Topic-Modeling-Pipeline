# stage_4_analysis.py
# (Replaces stage_4_spark_analysis.py)
# This is a 100% Spark-free, pure Python/Pandas analysis pipeline.

import torch
import pandas as pd
# --- NO MORE SPARK IMPORTS ---
from sklearn.cluster import KMeans as SklearnKMeans
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os
import sys
from tqdm import tqdm
import config
import utils

# --- 1. Global Caching for Summarizer (Unchanged) ---
model_cache = {}


def get_summarizer_pipeline():
    """
    Loads and caches the BART summarization pipeline.
    Uses GPU (device=0) and FP16 (float16) for optimization.
    """
    if "summarizer" not in model_cache:
        print("--- Caching: Loading BART summarization model (distilbart-cnn-12-6) to GPU with FP16 ---")
        
        if not torch.cuda.is_available():
            print("--- WARNING: CUDA not available. Loading model on CPU. This will be very slow. ---")
            model_cache["summarizer"] = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6"
            )
        else:
            model_cache["summarizer"] = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=0,
                torch_dtype=torch.float16
            )
    return model_cache["summarizer"]


# --- 2. Main Analysis Function (Pure Pandas) ---
def run_global_analysis():
    """
    Runs the entire pipeline.
    Uses Pandas/SQLAlchemy to load/save and Pandas/SKlearn for all AI.
    """
    print("--- Starting Stage 4: Global Analysis (Pure Pandas/SKlearn) ---")
    
    try:
        engine = utils.get_sqlalchemy_engine()
    except Exception as e:
        print("--- 🚨 FATAL ERROR: Could not create SQLAlchemy engine ---")
        print(f"Error: {e}")
        return False
    
    try:
        # --- Load Data from Postgres ---
        print("Loading all posts from PostgreSQL into Pandas...")
        sql_query = "SELECT post_id, cleaned_content FROM posts WHERE cleaned_content IS NOT NULL AND cleaned_content != ''"
        pd_posts = pd.read_sql(sql_query, engine)
        
        if len(pd_posts) < 20:  # Need a minimum for clustering
            print(f"Not enough posts ({len(pd_posts)}) to analyze. Aborting.")
            return False
        
        print(f"Loaded {len(pd_posts)} posts.")
        
        # --- Step 2: Embedding (in-driver) ---
        print("Loading sentence-transformer model (in-driver)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        print("Generating embeddings (in-driver)...")
        embeddings = sbert_model.encode(
            pd_posts['cleaned_content'].tolist(),
            batch_size=16,
            show_progress_bar=True
        )
        
        # --- Step 3: Clustering (in-driver with scikit-learn) ---
        NUM_TOPICS = 20
        print(f"Clustering posts into {NUM_TOPICS} topics with scikit-learn K-Means...")
        kmeans = SklearnKMeans(n_clusters=NUM_TOPICS, random_state=0, n_init=10)
        pd_posts['topic_id'] = kmeans.fit_predict(embeddings)
        
        # --- Step 4: Summarization (in-driver) ---
        print("Summarizing topics (in-driver)...")
        summarizer = get_summarizer_pipeline()
        
        topic_summaries = []
        
        for topic_id, group_df in tqdm(pd_posts.groupby('topic_id'), total=NUM_TOPICS):
            full_text = " . ".join(group_df['cleaned_content'].tolist())
            truncated_text = full_text[:4500]
            
            try:
                summary = summarizer(truncated_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            except Exception as e:
                print(f"Error summarizing topic {topic_id}: {e}")
                summary = f"Error generating summary: {e}"
            
            topic_summaries.append((int(topic_id), summary))
        
        pd_summaries = pd.DataFrame(topic_summaries, columns=["topic_id", "summary_text"])
        pd_mappings = pd_posts[['post_id', 'topic_id']]
        
        # --- Step 5: Save Results Back to DB ---
        print("Analysis complete. Saving results to PostgreSQL...")
        
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                print("Truncating old analysis results...")
                cur.execute("TRUNCATE TABLE global_topics, post_topic_mapping RESTART IDENTITY CASCADE;")
                conn.commit()
        
        # Save new results using Pandas to_sql
        pd_summaries.to_sql("global_topics", engine, if_exists='append', index=False)
        pd_mappings.to_sql("post_topic_mapping", engine, if_exists='append', index=False)
        
        print("Successfully saved global topics and post mappings.")
        
        print("\n--- Final Topic Summaries ---")
        print(pd_summaries.sort_values(by='topic_id').to_string())
        return True
    
    except Exception as e:
        print(f"\n--- 🚨 ERROR during Pandas AI Analysis ---")
        print(f"Error details: {e}")
        return False
    finally:
        if 'engine' in locals():
            engine.dispose()
        print("Analysis process finished.")


# --- Database Setup Function (Rewritten to be standalone) ---
def setup_database_tables():
    """
    Creates/Re-creates the tables for this new analysis pipeline.
    """
    print("Setting up database tables for global analysis...")
    
    CREATE_GLOBAL_TOPICS_SQL = """
                               DROP TABLE IF EXISTS post_topic_mapping CASCADE;
                               DROP TABLE IF EXISTS user_topics CASCADE;
                               DROP TABLE IF EXISTS global_topics CASCADE;

                               CREATE TABLE global_topics \
                               ( \
                                   topic_id     INTEGER PRIMARY KEY, \
                                   summary_text TEXT
                               );

                               CREATE TABLE post_topic_mapping \
                               ( \
                                   post_id  TEXT REFERENCES posts (post_id) ON DELETE CASCADE, \
                                   topic_id INTEGER REFERENCES global_topics (topic_id) ON DELETE CASCADE, \
                                   PRIMARY KEY (post_id, topic_id)
                               ); \
                               """
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_GLOBAL_TOPICS_SQL)
                conn.commit()
        print("Successfully dropped old tables and created 'global_topics' and 'post_topic_mapping' tables.")
    except Exception as e:
        print(f"--- 🚨 ERROR: Could not set up database tables ---")
        print(f"Error details: {e}")


if __name__ == "__main__":
    setup_database_tables()
    run_global_analysis()
