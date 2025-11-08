# stage_3_etl.py

import psycopg2
import pandas as pd
import re
import config
import utils

# --- MODIFIED: Added 'subreddits' column ---
CREATE_TABLES_SQL = """
                    CREATE TABLE IF NOT EXISTS users \
                    ( \
                        user_id    TEXT PRIMARY KEY, \
                        username   TEXT, \
                        personas   TEXT[], \
                        subreddits TEXT[] -- ADDED
                    );
                    CREATE TABLE IF NOT EXISTS follows \
                    ( \
                        follower_id TEXT REFERENCES users (user_id), \
                        followed_id TEXT REFERENCES users (user_id), \
                        PRIMARY KEY (follower_id, followed_id)
                    );
                    CREATE TABLE IF NOT EXISTS posts \
                    ( \
                        post_id         TEXT PRIMARY KEY, \
                        author_id       TEXT REFERENCES users (user_id), \
                        content         TEXT, \
                        cleaned_content TEXT, \
                        created_at      TIMESTAMP
                    ); \
                    """
# ---------------------------------------------

# (TRUNCATE_ALL_SQL and TRUNCATE_STAGE3_SQL are unchanged from your working version)
TRUNCATE_ALL_SQL = """
                   TRUNCATE TABLE
                       users,
                       follows,
                       posts,
                       user_topics,
                       post_topic_mapping
                       RESTART IDENTITY CASCADE; \
                   """

TRUNCATE_STAGE3_SQL = """
                      TRUNCATE TABLE
                          users,
                          follows,
                          posts
                          RESTART IDENTITY CASCADE; \
                      """


def create_tables():
    print("--- Stage 3.A: Creating Core Tables ---")
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLES_SQL)
                conn.commit()
        print("Successfully created 'users', 'follows', and 'posts' tables.")
    except Exception as e:
        print(f"--- 🚨 ERROR: Could not create tables ---")
        print(f"Error details: {e}")


def run_etl():
    print("\n--- Starting Stage 3.B: Pandas ETL ---")
    
    success, msg = utils.check_file_prerequisites(3)
    if not success:
        print(f"🚨 ERROR: {msg}")
        return False
    
    print("Stage 3.B.1: Truncating old data...")
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(TRUNCATE_ALL_SQL)
                except psycopg2.errors.UndefinedTable:
                    conn.rollback()
                    print("  (Note: Stage 4 tables not found. Truncating Stage 3 tables only.)")
                    cur.execute(TRUNCATE_STAGE3_SQL)
                conn.commit()
        print("Successfully truncated tables.")
    except Exception as e:
        print(f"--- 🚨 ERROR: Could not truncate tables ---")
        print(f"Error details: {e}")
        return False
    
    print("Stage 3.B.2: Initializing SQLAlchemy engine...")
    try:
        engine = utils.get_sqlalchemy_engine()
        
        # --- USERS (No change needed) ---
        # Pandas and SQLAlchemy are smart enough to handle the new
        # 'subreddits' column (which is a list -> TEXT[]) automatically.
        print("Processing 'users' data...")
        df_users = pd.read_json(config.USERS_JSON_PATH, orient='records')
        df_users.to_sql("users", engine, if_exists='append', index=False)
        print(f"Successfully loaded {len(df_users)} users.")
        
        # --- FOLLOWS (Unchanged) ---
        print("Processing 'follows' data...")
        df_follows = pd.read_json(config.FOLLOWS_JSON_PATH, orient='records')
        df_follows.to_sql("follows", engine, if_exists='append', index=False)
        print(f"Successfully loaded {len(df_follows)} follows.")
        
        # --- POSTS (Unchanged) ---
        print("Processing 'posts' data (with Pandas)...")
        df_posts = pd.read_json(config.POSTS_JSON_PATH, orient='records')
        
        cleaned_content = df_posts['content'].str.lower()
        cleaned_content = cleaned_content.str.replace(r'http\S+', '', regex=True)
        cleaned_content = cleaned_content.str.replace(r'[^a-z0-9\s]', '', regex=True)
        cleaned_content = cleaned_content.str.replace(r'\s+', ' ', regex=True)
        cleaned_content = cleaned_content.str.strip()
        cleaned_content = cleaned_content.replace('', None)
        
        df_posts['cleaned_content'] = cleaned_content
        df_posts['created_at'] = pd.to_datetime(df_posts['created_at'])
        
        df_posts.to_sql("posts", engine, if_exists='append', index=False)
        print(f"Successfully loaded {len(df_posts)} posts.")
        
        print("\n--- Pandas ETL Job Complete! ---")
        return True
    
    except Exception as e:
        print(f"\n--- 🚨 ERROR during Pandas ETL ---")
        print(f"Error details: {e}")
        return False
    finally:
        if 'engine' in locals():
            engine.dispose()
        print("ETL process finished.")


if __name__ == "__main__":
    create_tables()
    run_etl()
