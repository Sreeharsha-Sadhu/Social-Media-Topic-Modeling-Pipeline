# stage_4_analysis.py

import psycopg2
import pandas as pd
from bertopic import BERTopic
from transformers import pipeline
from psycopg2.extras import execute_values
from tqdm import tqdm
import sys
import config
import utils

# SQL to create the results tables
CREATE_RESULTS_TABLES_SQL = """
                            CREATE TABLE IF NOT EXISTS user_topics \
                            ( \
                                topic_id \
                                SERIAL \
                                PRIMARY \
                                KEY, \
                                user_id \
                                TEXT \
                                REFERENCES \
                                users \
                            ( \
                                user_id \
                            ) ON DELETE CASCADE,
                                topic_name TEXT,
                                summary TEXT,
                                post_count INTEGER,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                );

                            CREATE TABLE IF NOT EXISTS post_topic_mapping \
                            ( \
                                post_id \
                                TEXT \
                                REFERENCES \
                                posts \
                            ( \
                                post_id \
                            ) ON DELETE CASCADE,
                                topic_id INTEGER REFERENCES user_topics \
                            ( \
                                topic_id \
                            ) \
                              ON DELETE CASCADE,
                                PRIMARY KEY \
                            ( \
                                post_id, \
                                topic_id \
                            )
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
        # By setting ON DELETE CASCADE, we only need to delete from the parent table
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


def run_analysis_pipeline(user_id):
    """Main function to run the full AI analysis pipeline for a user."""
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
            return True  # Not a failure, just no-op
        
        posts_list = feed_df['cleaned_content'].tolist()
        
        print("Initializing BERTopic model...")
        topic_model = BERTopic(
            embedding_model="all-MiniLM-L6-v2",
            min_topic_size=5,
            verbose=False
        )
        
        print("Initializing Summarization model (this may take a moment)...")
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-6-6",
            truncation=True
        )
        
        print("Running BERTopic. This may take a few minutes...")
        topics, _ = topic_model.fit_transform(posts_list)
        feed_df['topic'] = topics
        topic_info_df = topic_model.get_topic_info()
        print(f"BERTopic found {len(topic_info_df) - 1} topics.")
        
        print("Summarizing topics and saving results...")
        all_post_topic_mappings = []
        
        for _, topic_row in tqdm(topic_info_df.iterrows(), total=topic_info_df.shape[0]):
            topic_num = topic_row['Topic']
            topic_name = topic_row['Name']
            if topic_num == -1:
                continue
            
            topic_posts_df = feed_df[feed_df['topic'] == topic_num]
            topic_posts = topic_posts_df['cleaned_content'].tolist()
            post_count = len(topic_posts)
            doc_to_summarize = " . ".join(topic_posts)
            
            max_chars = 4000
            if len(doc_to_summarize) > max_chars:
                doc_to_summarize = doc_to_summarize[:max_chars]
            
            summary_result = summarizer(doc_to_summarize, min_length=15, max_length=50)
            summary_text = summary_result[0]['summary_text']
            
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_topics (user_id, topic_name, summary, post_count)
                    VALUES (%s, %s, %s, %s) RETURNING topic_id;
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
    # Allows running: python stage_4_analysis.py user_3
    if len(sys.argv) > 1:
        user_id_to_run = sys.argv[1]
        create_results_tables()  # Ensure tables exist
        run_analysis_pipeline(user_id_to_run)
    else:
        print("Usage: python stage_4_analysis.py <user_id>")
        print("Example: python stage_4_analysis.py user_1")
