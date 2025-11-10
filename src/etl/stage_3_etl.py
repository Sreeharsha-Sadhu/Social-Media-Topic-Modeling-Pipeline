# stage_3_etl.py
"""
Stage 3: ETL (Dual-mode: Pandas or Spark)

This file preserves your original Pandas ETL implementation and adds
a Spark-based ETL path controlled by the environment flag:
    USE_SPARK_ETL=true/false

Make sure you've added these lines to your .env or environment:
    USE_SPARK_ETL=true
    SPARK_MASTER=local[*]
    SPARK_APP_NAME=SocialMediaETL
"""

import pandas as pd
import psycopg2
from sqlalchemy.types import ARRAY, String

from src.common import utils
from src.config import settings

# --- Existing SQL (unchanged) ---
CREATE_TABLES_SQL = """
                    DROP TABLE IF EXISTS posts CASCADE;
                    DROP TABLE IF EXISTS follows CASCADE;
                    DROP TABLE IF EXISTS users CASCADE;

                    CREATE TABLE IF NOT EXISTS users \
                    ( \
                        user_id    TEXT PRIMARY KEY, \
                        username   TEXT, \
                        personas   TEXT[], \
                        subreddits TEXT[] \
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


# -------------------------
# Helper: Spark initialization (tries utils.get_spark_session if present)
# -------------------------
def _init_spark():
    """
    Return a SparkSession. Prefer utils.get_spark_session() if defined.
    If it's not available, attempt to create a local SparkSession using config flags.
    """
    try:
        # Prefer expected helper in utils (if you added it)
        get_spark_session = getattr(utils, "get_spark_session", None)
        if callable(get_spark_session):
            spark = get_spark_session()
            return spark
    except Exception:
        # Fall through to local creation
        pass
    
    # Try to import pyspark and create a session locally
    try:
        from pyspark.sql import SparkSession
        print(f"🧩 Initializing SparkSession ({settings.SPARK_APP_NAME}) master={settings.SPARK_MASTER} ...")
        spark = (
            SparkSession.builder
            .appName(settings.SPARK_APP_NAME if hasattr(settings, "SPARK_APP_NAME") else "SocialMediaETL")
            .master(settings.SPARK_MASTER if hasattr(settings, "SPARK_MASTER") else "local[*]")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.repl.eagerEval.enabled", "true")
            .getOrCreate()
        )
        return spark
    except Exception as e:
        raise RuntimeError(f"Could not initialize SparkSession: {e}")


# -------------------------
# Pandas ETL (original logic moved to a helper)
# -------------------------
def _run_etl_pandas():
    """Original Pandas ETL path (keeps your existing logic)."""
    print("--- Running Pandas ETL ---")
    try:
        engine = utils.get_sqlalchemy_engine()
        
        # --- USERS ---
        print("Processing 'users' data...")
        df_users = pd.read_json(settings.USERS_JSON_PATH, orient='records')
        
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
        df_follows = pd.read_json(settings.FOLLOWS_JSON_PATH, orient='records')
        df_follows.to_sql("follows", engine, if_exists='append', index=False)
        print(f"Successfully loaded {len(df_follows)} follows.")
        
        # --- POSTS ---
        print("Processing 'posts' data (with Pandas)...")
        df_posts = pd.read_json(settings.POSTS_JSON_PATH, orient='records')
        
        # Cleaning (original Pandas code)
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
        try:
            if 'engine' in locals():
                engine.dispose()
        except Exception:
            pass
        print("ETL process finished (Pandas).")


# -------------------------
# Spark ETL (safe: no Python UDFs)
# -------------------------
def _run_etl_spark():
    """
    Spark-based ETL path (uses Spark native transforms and JDBC writes).
    Avoids Python UDFs to prevent serialization/worker issues.
    """
    print("--- 🚀 Running Spark ETL ---")
    try:
        try:
            spark = _init_spark()
        except RuntimeError as e:
            print(f"🚨 Spark initialization failed: {e}")
            print("Falling back to Pandas ETL.")
            return _run_etl_pandas()
        
        from pyspark.sql import functions as F
        
        jdbc_url = f"jdbc:postgresql://{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        jdbc_opts = {
            "url": jdbc_url,
            "user": settings.DB_USER,
            "password": settings.DB_PASS,
            "driver": "org.postgresql.Driver"
        }
        
        # --- USERS ---
        print("Reading users.json via Spark...")
        try:
            df_users = spark.read.option("multiLine", "true").json(settings.USERS_JSON_PATH)
        except Exception as e:
            print(f"🚨 Failed to read users JSON with Spark: {e}")
            raise
        
        print("Writing 'users' table to PostgreSQL via JDBC (append)...")
        (df_users.write
         .format("jdbc")
         .option("url", jdbc_opts["url"])
         .option("dbtable", "users")
         .option("user", jdbc_opts["user"])
         .option("password", jdbc_opts["password"])
         .option("driver", jdbc_opts["driver"])
         .mode("append")
         .save()
         )
        
        # --- FOLLOWS ---
        print("Reading follows.json via Spark...")
        df_follows = spark.read.option("multiLine", "true").json(settings.FOLLOWS_JSON_PATH)
        
        print("Writing 'follows' table to PostgreSQL via JDBC (append)...")
        (df_follows.write
         .format("jdbc")
         .option("url", jdbc_opts["url"])
         .option("dbtable", "follows")
         .option("user", jdbc_opts["user"])
         .option("password", jdbc_opts["password"])
         .option("driver", jdbc_opts["driver"])
         .mode("append")
         .save()
         )
        
        # --- POSTS ---
        print("Reading posts.json via Spark...")
        df_posts = spark.read.option("multiLine", "true").json(settings.POSTS_JSON_PATH)
        
        print("Cleaning post text content (Spark native functions, no UDFs)...")
        df_posts = df_posts.withColumn("cleaned_content", F.lower(F.col("content")))
        df_posts = df_posts.withColumn("cleaned_content", F.regexp_replace("cleaned_content", r"http\S+", ""))
        # Note: allow digits and letters; remove other punctuation
        df_posts = df_posts.withColumn("cleaned_content", F.regexp_replace("cleaned_content", r"[^a-z0-9\s]", ""))
        df_posts = df_posts.withColumn("cleaned_content", F.regexp_replace("cleaned_content", r"\s+", " "))
        df_posts = df_posts.withColumn("cleaned_content", F.trim(F.col("cleaned_content")))
        # Convert created_at to timestamp (assuming ISO-like strings)
        df_posts = df_posts.withColumn("created_at", F.to_timestamp("created_at"))
        
        print("Writing 'posts' table to PostgreSQL via JDBC (append)...")
        (df_posts.write
         .format("jdbc")
         .option("url", jdbc_opts["url"])
         .option("dbtable", "posts")
         .option("user", jdbc_opts["user"])
         .option("password", jdbc_opts["password"])
         .option("driver", jdbc_opts["driver"])
         .mode("append")
         .save()
         )
        
        print("\n✅ Spark ETL Complete!")
        return True
    
    except Exception as e:
        print(f"\n--- 🚨 Spark ETL Error ---\nError details: {e}")
        print("Falling back to Pandas ETL as a safety measure.")
        try:
            return _run_etl_pandas()
        except Exception as e2:
            print(f"Also failed to run Pandas ETL: {e2}")
            return False
    finally:
        # Don't stop shared Spark sessions if created by utils; only stop if we created it here.
        try:
            # If utils has get_spark_session, don't stop it (singleton). Otherwise attempt stop.
            if hasattr(utils, "get_spark_session"):
                # Assume utils manages lifecycle
                pass
            else:
                # Best-effort stop
                try:
                    spark.stop()
                except Exception:
                    pass
        except Exception:
            pass
        print("ETL process finished (Spark).")


# -------------------------
# Public entrypoint: run_etl (keeps previous behavior but chooses branch)
# -------------------------
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
    """Runs the full ETL pipeline: Truncate and then run selected (Spark/Pandas) implementation."""
    print("\n--- Starting Stage 3.B: ETL ---")
    mode = "Spark" if getattr(settings, "USE_SPARK_ETL", False) else "Pandas"
    print(f"ETL Mode: {mode}")
    
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
    
    # Route to Spark or Pandas path
    if getattr(settings, "USE_SPARK_ETL", False):
        return _run_etl_spark()
    else:
        return _run_etl_pandas()


# If run as script
if __name__ == "__main__":
    create_tables()
    run_etl()
