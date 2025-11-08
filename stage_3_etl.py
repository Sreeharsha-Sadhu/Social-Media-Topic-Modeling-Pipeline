# stage_3_etl.py

import psycopg2
import pandas as pd
import re
import config
import utils
from sqlalchemy.types import ARRAY, String

# --- MODIFIED SECTION ---
# We add DROP TABLE commands to force the schema to update
# when 'Setup Database' is run.
CREATE_TABLES_SQL = """
                    DROP TABLE IF EXISTS posts CASCADE;
                    DROP TABLE IF EXISTS follows CASCADE;
                    DROP TABLE IF EXISTS users CASCADE;

                    CREATE TABLE IF NOT EXISTS users \
                    ( \
                        user_id    TEXT PRIMARY KEY, \
                        username   TEXT, \
                        personas   TEXT[], \
                        subreddits TEXT[]
                    );
                    CREATE TABLE IF NOT EXISTS follows \
                    ( \
                        follower_id TEXT REFERENCES users (user_id) ON DELETE CASCADE, \
                        followed_id TEXT REFERENCES users (user_id) ON DELETE CASCADE, \
                        PRIMARY KEY (follower_id, followed_id)
                    );
                    CREATE TABLE IF NOT EXISTS posts \
                    ( \
                        post_id         TEXT PRIMARY KEY, \
                        author_id       TEXT REFERENCES users (user_id) ON DELETE CASCADE, \
                        content         TEXT, \
                        cleaned_content TEXT, \
                        created_at      TIMESTAMP
                    ); \
                    """
# --- END MODIFIED SECTION ---

# TRUNCATE SQL (unchanged)
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
    """Creates the core DB tables. Now DESTRUCTIVE."""
    print("--- Stage 3.A: Forcing recreation of Core Tables ---")
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
    """Runs the full ETL pipeline: Truncate and load with Pandas."""
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
        
        # --- USERS ---
        print("Processing 'users' data...")
        df_users = pd.read_json(config.USERS_JSON_PATH, orient='records')
        
        # This dtype mapping is correct and necessary
        df_users.to_sql(
            "users",
            engine,
            if_exists='append',
            index=False,
            dtype={'personas': ARRAY(String), 'subreddits': ARRAY(String)}
        )
        
        print(f"Successfully loaded {len(df_users)} users.")
        
        # --- FOLLOWS ---
        print("Processing 'follows' data...")
        df_follows = pd.read_json(config.FOLLOWS_JSON_PATH, orient='records')
        df_follows.to_sql("follows", engine, if_exists='append', index=False)
        print(f"Successfully loaded {len(df_follows)} follows.")
        
        # --- POSTS ---
        print("Processing 'posts' data (with Pandas)...")
        df_posts = pd.read_json(config.POSTS_JSON_PATH, orient='records')
        
        cleaned_content = df_posts['content'].str.lower()
        cleaned_content = cleaned_content.str.replace(r'http\S+', '', regex=True)
        cleaned_content = cleaned_content.str.replace(r'[^a-z0_9\s]', '', regex=True)
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
