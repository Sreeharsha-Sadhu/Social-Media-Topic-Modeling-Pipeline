# stage_4_analysis.py

import psycopg2
import pandas as pd
from bertopic import BERTopic
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from psycopg2.extras import execute_values
from tqdm import tqdm
import sys
import config
import utils

# SQL to create the results tables (unchanged)
CREATE_RESULTS_TABLES_SQL = """
                            CREATE TABLE IF NOT EXISTS user_topics \
                            ( \
                                topic_id   SERIAL PRIMARY KEY, \
                                user_id    TEXT REFERENCES users (user_id) ON DELETE CASCADE, \
                                topic_name TEXT, \
                                summary    TEXT, \
                                post_count INTEGER, \
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            );

                            CREATE TABLE IF NOT EXISTS post_topic_mapping \
                            ( \
                                post_id  TEXT REFERENCES posts (post_id) ON DELETE CASCADE, \
                                topic_id INTEGER REFERENCES user_topics (topic_id) ON DELETE CASCADE, \
                                PRIMARY KEY (post_id, topic_id)
                            ); \
                            """


def create_results_tables():
    """Creates the result/analysis DB tables if they don't exist."""
    print("--- Stage 4.A: Creating Results Tables ---")
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_RESULTS_TABLES_SQL)
                conn.commit()
        print("Successfully created 'user_topics' and 'post_topic_mapping' tables.")
    except Exception as e:
        print(f"--- 🚨 ERROR: Could not create results tables ---")
        print(f"Error details: {e}")


def clear_previous_analysis(conn, user_id):
    """Clears out old analysis results for a user."""
    print(f"Cleared old analysis results for {user_id}.")
    with conn.cursor() as cur:
        cur.execute("DELETE FROM user_topics WHERE user_id = %s", (user_id,))
        conn.commit()


def get_feed_for_user(conn, user_id):
    """Fetches the unique feed for a given user."""
    print(f"Fetching feed for {user_id}...")
    sql_query = """
                SELECT p.post_id, p.cleaned_content
                FROM posts p
                         JOIN follows f ON p.author_id = f.followed_id
                WHERE f.follower_id = %s
                  AND p.cleaned_content IS NOT NULL
                  AND p.cleaned_content != ''; \
                """
    with conn.cursor() as cur:
        cur.execute(sql_query, (user_id,))
        rows = cur.fetchall()
    
    df = pd.DataFrame(rows, columns=['post_id', 'cleaned_content'])
    
    if df.empty:
        print(f"No posts found for {user_id}'s feed.")
        return None
    
    print(f"Found {len(df)} posts in {user_id}'s feed.")
    return df


# --- 3. AI Core (Topic Modeling & Summarization) ---
def run_analysis_pipeline(user_id):
    """
    Main function to run the full AI analysis pipeline for a user.
    """
    conn = None
    try:
        conn = utils.get_db_connection()
    except Exception as e:
        print("--- 🚨 ERROR: Could not connect to database. ---")
        print("Please ensure PostgreSQL is running and config.py is correct.")
        print(f"Error details: {e}")
        return False
    
    try:
        clear_previous_analysis(conn, user_id)
        
        feed_df = get_feed_for_user(conn, user_id)
        if feed_df is None or len(feed_df) < 10:
            print(f"Not enough posts ({0 if feed_df is None else len(feed_df)}) to analyze. Aborting.")
            return True
        
        posts_list = feed_df['cleaned_content'].tolist()
        
        print("Initializing BERTopic model...")
        topic_model = BERTopic(
            embedding_model="all-MiniLM-L6-v2",
            min_topic_size=5,
            verbose=False
        )
        
        print("Initializing FLAN-T5 model (this may take a few minutes on first run)...")
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        print("Running BERTopic. This may take a few minutes...")
        topics, _ = topic_model.fit_transform(posts_list)
        feed_df['topic'] = topics
        topic_info_df = topic_model.get_topic_info()
        print(f"BERTopic found {len(topic_info_df) - 1} topics.")
        
        print("Generating summaries and titles for each topic...")
        all_post_topic_mappings = []
        
        for _, topic_row in tqdm(topic_info_df.iterrows(), total=topic_info_df.shape[0]):
            topic_num = topic_row['Topic']
            
            if topic_num == -1:
                continue
            
            topic_posts_df = feed_df[feed_df['topic'] == topic_num]
            topic_posts = topic_posts_df['cleaned_content'].tolist()
            post_count = len(topic_posts)
            
            doc_to_summarize = " . ".join(topic_posts)
            
            # --- MODIFIED SECTION: Two-Pass Generation ---
            
            # --- 1. Create the Summary (with new settings) ---
            summary_prompt = f"""
Read the following social media posts:
---
{doc_to_summarize[:4000]}
---
Write a single, coherent paragraph that summarizes the main topic being discussed.
SUMMARY:
"""
            input_ids = tokenizer(summary_prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids
            # INCREASED max_length to 150 and min_length to 30 to allow for fuller sentences
            outputs = model.generate(input_ids, max_length=150, min_length=30, num_beams=4, no_repeat_ngram_size=2)
            summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # --- 2. Create the Title (with new prompt and settings) ---
            # IMPROVED prompt to ask for a "human-readable" title
            title_prompt = f"""
Based on the following summary, what is a short, human-readable topic title?
---
{summary_text}
---
A descriptive 3-to-5 word title is:
"""
            input_ids = tokenizer(title_prompt, return_tensors="pt", truncation=True, max_length=256).input_ids
            # INCREASED max_length to 30 to give it more room for a natural title
            outputs = model.generate(input_ids, max_length=30, min_length=3, num_beams=2)
            topic_name = tokenizer.decode(outputs[0], skip_special_tokens=True).replace("TITLE:", "").strip()
            
            # --- END MODIFIED SECTION ---
            
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_topics (user_id, topic_name, summary, post_count)
                    VALUES (%s, %s, %s, %s)
                    RETURNING topic_id;
                    """,
                    (user_id, topic_name, summary_text, post_count)
                )
                new_topic_id = cur.fetchone()[0]
                conn.commit()
            
            for post_id in topic_posts_df['post_id'].tolist():
                all_post_topic_mappings.append((post_id, new_topic_id))
        
        if all_post_topic_mappings:
            print(f"Saving {len(all_post_topic_mappings)} post-topic mappings...")
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    "INSERT INTO post_topic_mapping (post_id, topic_id) VALUES %s",
                    all_post_topic_mappings
                )
                conn.commit()
        
        print("\n--- Stage 4: Analysis Complete! ---")
        print(f"Successfully analyzed and saved results for {user_id}.")
        return True
    
    except (Exception, psycopg2.Error) as error:
        print(f"--- 🚨 ERROR in Stage 4 ---")
        print(f"Error details: {error}")
        if conn:
            conn.rollback()
        return False
    
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_id_to_run = sys.argv[1]
        create_results_tables()
        run_analysis_pipeline(user_id_to_run)
    else:
        print("Usage: python stage_4_analysis.py <user_id>")
        print("Example: python stage_4_analysis.py user_1")
