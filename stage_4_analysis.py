# stage_4_analysis.py
# (Replaces stage_4_spark_analysis.py)
# This version fixes all warnings and improves the prompt.

import torch
import pandas as pd
from sklearn.cluster import KMeans as SklearnKMeans
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os
import sys
from tqdm import tqdm
import config
import utils

# --- 1. Global Caching for LLM ---
model_cache = {}


def get_llm_pipeline():
    """
    Loads and caches the FLAN-T5-Large text-generation pipeline.
    """
    if "llm" not in model_cache:
        print("--- Caching: Loading FLAN-T5-Large model (770M params) to GPU with FP16 ---")
        print("   (This may take several minutes and >3GB of VRAM on first run)")
        
        model_name = "google/flan-t5-large"
        
        # --- FIX: Use dtype=torch.float16 ---
        if not torch.cuda.is_available():
            print("--- WARNING: CUDA not available. Loading model on CPU. This will be VERY slow. ---")
            model_cache["llm"] = pipeline(
                "text2text-generation",
                model=model_name
            )
        else:
            model_cache["llm"] = pipeline(
                "text2text-generation",
                model=model_name,
                device=0,
                dtype=torch.float16  # Fixed 'torch_dtype' warning
            )
    return model_cache["llm"]


# --- 2. Main Analysis Function (Upgraded) ---
def run_global_analysis():
    """
    Runs the entire pipeline with fixed prompts and settings.
    """
    print("--- Starting Stage 4: Global Analysis (FLAN-T5-Large) ---")
    
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
        
        if len(pd_posts) < 20:
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
        
        # --- Step 4: Summarization (with new model and prompt) ---
        print("Summarizing topics (in-driver with FLAN-T5-Large)...")
        generator = get_llm_pipeline()
        
        topic_summaries = []
        
        for topic_id, group_df in tqdm(pd_posts.groupby('topic_id'), total=NUM_TOPICS):
            
            full_text = " . ".join(group_df['cleaned_content'].tolist())
            
            # --- NEW PROMPT ---
            # This prompt is more direct and content-focused.
            prompt = f"""
Instructions: Read the following social media posts and write a concise, abstractive summary. The summary must be a coherent paragraph that captures the main topic and sentiment of the discussion. Do not just copy and paste sentences.

Posts:
{full_text}

Summary:
"""
            
            try:
                # --- NEW GENERATOR CALL ---
                # This call fixes all warnings:
                # 1. truncation=True: Truncates the input prompt
                # 2. max_length=512: Sets the *input* limit, fixing the (1036 > 512) warning
                # 3. max_new_tokens=150: Sets the *output* limit, fixing the conflict warning
                summary = generator(
                    prompt,
                    truncation=True,
                    max_length=512,
                    max_new_tokens=150,
                    min_length=30,
                    do_sample=False,
                    no_repeat_ngram_size=2
                )[0]['generated_text']
            
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


# --- Database Setup Function (Unchanged) ---
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
