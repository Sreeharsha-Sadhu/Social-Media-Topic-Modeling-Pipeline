"""
stage_3_etl.py (unified)
──────────────────────────────────────────────────────
Unified ETL for:
 - Synthetic ingestion (users, follows, posts) using Pandas or Spark
 - Live feed persistence (reddit / linkedin / twitter): caching, storing summaries & topic outputs

Key functions:
 - create_tables()    : (Destructive for core synthetic tables) also ensures live tables exist
 - run_etl()          : runs synthetic ETL (Pandas or Spark per config)
 - ingest_live_cache(...)  : persist a live analysis run (raw posts, topic rows, summary)
 - get_last_live_cache(user_id, platform) : returns latest cached result for platform/user
 - record_live_run(...) : light run logging
──────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd
from pyspark.sql import functions as F
from sqlalchemy.types import ARRAY, String

from src.core import config, utils
from src.core.logging_config import get_logger

logger = get_logger(__name__)

# existing SQL used for synthetic tables (unchanged)
CREATE_TABLES_SQL = """
                    DROP TABLE IF EXISTS posts CASCADE;
                    DROP TABLE IF EXISTS follows CASCADE;
                    DROP TABLE IF EXISTS users CASCADE;

                    CREATE TABLE IF NOT EXISTS users
                    (
                        user_id    TEXT PRIMARY KEY,
                        username   TEXT,
                        personas   TEXT[],
                        subreddits TEXT[]
                    );

                    CREATE TABLE IF NOT EXISTS follows
                    (
                        follower_id TEXT REFERENCES users (user_id) ON DELETE CASCADE,
                        followed_id TEXT REFERENCES users (user_id) ON DELETE CASCADE,
                        PRIMARY KEY (follower_id, followed_id)
                    );

                    CREATE TABLE IF NOT EXISTS posts
                    (
                        post_id         TEXT PRIMARY KEY,
                        author_id       TEXT REFERENCES users (user_id) ON DELETE CASCADE,
                        content         TEXT,
                        cleaned_content TEXT,
                        created_at      TIMESTAMP
                    ); \
                    """

# Truncate core synthetic tables (used by run_etl)
TRUNCATE_STAGE3_SQL = "TRUNCATE TABLE users, follows, posts RESTART IDENTITY CASCADE;"

# SQL snippets for live tables (created by create_tables as non-destructive "ensure exists")
LIVE_CACHE_SQL = """
                 CREATE TABLE IF NOT EXISTS live_feed_runs
                 (
                     run_id     SERIAL PRIMARY KEY,
                     user_id    TEXT      REFERENCES users (user_id) ON DELETE SET NULL,
                     platform   TEXT      NOT NULL,
                     run_status TEXT      NOT NULL,
                     fetched_at TIMESTAMP NOT NULL DEFAULT NOW(),
                     notes      TEXT
                 );

                 CREATE TABLE IF NOT EXISTS live_feed_cache
                 (
                     cache_id    SERIAL PRIMARY KEY,
                     user_id     TEXT REFERENCES users (user_id) ON DELETE CASCADE,
                     platform    TEXT      NOT NULL,
                     fetched_at  TIMESTAMP NOT NULL DEFAULT NOW(),
                     raw_posts   JSONB,
                     summary     TEXT,
                     topic_data  JSONB,
                     source_meta JSONB,
                     UNIQUE (user_id, platform, fetched_at)
                 );
                 CREATE INDEX IF NOT EXISTS idx_live_cache_user_platform_fetched_at
                     ON live_feed_cache (user_id, platform, fetched_at DESC); \
                 """


# -------------------------
# Spark initialization (prefers utils.get_spark_session)
# -------------------------
def _init_spark():
    try:
        if hasattr(utils, "get_spark_session"):
            return utils.get_spark_session()
    except Exception:
        pass
    
    from pyspark.sql import SparkSession
    logger.info(f"Initializing SparkSession for {config.SPARK_APP_NAME} ...")
    spark = (
        SparkSession.builder
        .appName(config.SPARK_APP_NAME)
        .master(config.SPARK_MASTER)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    logger.info("SparkSession started.")
    return spark


# -------------------------
# Pandas ETL
# -------------------------
def _run_etl_pandas() -> bool:
    logger.info("Running Pandas ETL ...")
    try:
        engine = utils.get_sqlalchemy_engine()
        
        # Users
        df_users = pd.read_json(config.USERS_JSON_PATH, orient="records")
        logger.info("Inserting users...")
        df_users.to_sql("users", engine, if_exists="append", index=False,
                        dtype={"personas": ARRAY(String), "subreddits": ARRAY(String)})
        
        # Follows
        df_follows = pd.read_json(config.FOLLOWS_JSON_PATH, orient="records")
        logger.info("Inserting follows...")
        df_follows.to_sql("follows", engine, if_exists="append", index=False)
        
        # Posts
        df_posts = pd.read_json(config.POSTS_JSON_PATH, orient="records")
        logger.info("Cleaning posts...")
        cleaned = (
            df_posts["content"]
            .str.lower()
            .str.replace(r"http\S+", "", regex=True)
            .str.replace(r"[^a-z0-9\s]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df_posts["cleaned_content"] = cleaned.replace("", None)
        df_posts["created_at"] = pd.to_datetime(df_posts["created_at"], errors="coerce")
        
        logger.info("Inserting posts...")
        df_posts.to_sql("posts", engine, if_exists="append", index=False)
        
        logger.info("Pandas ETL completed.")
        return True
    except Exception as e:
        logger.exception("Pandas ETL failed.")
        return False
    finally:
        if "engine" in locals():
            engine.dispose()


# -------------------------
# Spark ETL
# -------------------------
def _run_etl_spark() -> bool:
    logger.info("Running Spark ETL ...")
    try:
        spark = _init_spark()
        jdbc_url = f"jdbc:postgresql://{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
        jdbc_opts = {
            "url": jdbc_url,
            "user": config.DB_USER,
            "password": config.DB_PASS,
            "driver": "org.postgresql.Driver",
        }
        
        # Users
        logger.info("Loading users.json into Spark...")
        df_users = spark.read.option("multiLine", "true").json(str(config.USERS_JSON_PATH))
        df_users.write.format("jdbc").options(**jdbc_opts, dbtable="users").mode("append").save()
        logger.info("Users inserted via JDBC.")
        
        # Follows
        df_follows = spark.read.option("multiLine", "true").json(str(config.FOLLOWS_JSON_PATH))
        df_follows.write.format("jdbc").options(**jdbc_opts, dbtable="follows").mode("append").save()
        logger.info("Follows inserted via JDBC.")
        
        # Posts
        df_posts = spark.read.option("multiLine", "true").json(str(config.POSTS_JSON_PATH))
        
        df_posts = df_posts.withColumn("cleaned_content", F.lower(F.col("content")))
        df_posts = df_posts.withColumn("cleaned_content", F.regexp_replace("cleaned_content", r"http\\S+", ""))
        df_posts = df_posts.withColumn("cleaned_content", F.regexp_replace("cleaned_content", r"[^a-z0-9\\s]", ""))
        df_posts = df_posts.withColumn("cleaned_content", F.regexp_replace("cleaned_content", r"\\s+", " "))
        df_posts = df_posts.withColumn("cleaned_content", F.trim(F.col("cleaned_content")))
        
        # Normalize timestamps safely (best-effort)
        df_posts = df_posts.withColumn("created_at", F.regexp_replace("created_at", "Z$", ""))
        df_posts = df_posts.withColumn(
            "created_at",
            F.expr("try_to_timestamp(created_at, \"yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX\")")
        )
        
        df_posts.write.format("jdbc").options(**jdbc_opts, dbtable="posts").mode("append").save()
        logger.info("Posts inserted via JDBC.")
        
        logger.info("Spark ETL completed.")
        return True
    except Exception as e:
        logger.exception("Spark ETL failed.")
        if "CAST_INVALID_INPUT" in str(e) or "TIMESTAMP" in str(e):
            logger.warning("timestamp parse issue - falling back to pandas ETL")
            return _run_etl_pandas()
        return False
    finally:
        # Only stop if we created the session here (utils manages it otherwise)
        try:
            if not hasattr(utils, "get_spark_session"):
                spark.stop()
        except Exception:
            pass


# -------------------------
# Live ingestion helpers
# -------------------------
def record_live_run(user_id: Optional[str], platform: str, run_status: str, notes: Optional[str] = None) -> None:
    """
    Lightweight logging of a live-analysis run into live_feed_runs.
    run_status one of: 'success', 'cached', 'failed'
    """
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO live_feed_runs (user_id, platform, run_status, notes) VALUES (%s, %s, %s, %s);",
                    (user_id, platform, run_status, notes)
                )
                conn.commit()
    except Exception:
        logger.exception("Failed to record live run.")


def ingest_live_cache(user_id: str,
                      platform: str,
                      raw_posts: List[Dict[str, Any]],
                      topic_rows: List[Dict[str, Any]],
                      summary_text: Optional[str] = None,
                      source_meta: Optional[Dict[str, Any]] = None) -> bool:
    """
    Persist a live analysis output into live_feed_cache.
    raw_posts: list of dicts {post_id, content, created_at, ...}
    topic_rows: list of dicts [{topic_id, topic_title, summary, sample_posts}, ...]

    Returns True on success.
    """
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO live_feed_cache (user_id, platform, raw_posts, summary, topic_data, source_meta)
                    VALUES (%s, %s, %s::jsonb, %s, %s::jsonb)
                    RETURNING cache_id;
                    """,
                    (user_id,
                     platform,
                     json.dumps(raw_posts) if raw_posts is not None else None,
                     summary_text,
                     json.dumps(topic_rows) if topic_rows is not None else None)
                )
                cid = cur.fetchone()[0]
                conn.commit()
        logger.info("Live cache stored (cache_id=%s) for user=%s platform=%s", cid, user_id, platform)
        record_live_run(user_id, platform, "success", f"cache_id={cid}")
        return True
    except Exception:
        logger.exception("Failed to persist live cache.")
        record_live_run(user_id, platform, "failed", "ingest_live_cache error")
        return False


def get_last_live_cache(user_id: str, platform: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the latest live cache for (user_id, platform).
    Returns a dict with fields: cache_id, user_id, platform, fetched_at, raw_posts, summary, topic_data, source_meta
    or None if no cache exists.
    """
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT cache_id,
                           user_id,
                           platform,
                           fetched_at,
                           raw_posts::text,
                           summary,
                           topic_data::text,
                           source_meta::text
                    FROM live_feed_cache
                    WHERE user_id = %s
                      AND platform = %s
                    ORDER BY fetched_at DESC
                    LIMIT 1;
                    """,
                    (user_id, platform)
                )
                row = cur.fetchone()
                if not row:
                    return None
                cache_id, uid, platform, fetched_at, raw_posts_text, summary, topic_data_text, source_meta_text = row
                # parse JSON text back to Python objects
                raw_posts = json.loads(raw_posts_text) if raw_posts_text else []
                topic_data = json.loads(topic_data_text) if topic_data_text else []
                source_meta = json.loads(source_meta_text) if source_meta_text else None
                return {
                    "cache_id": cache_id,
                    "user_id": uid,
                    "platform": platform,
                    "fetched_at": fetched_at,
                    "raw_posts": raw_posts,
                    "summary": summary,
                    "topic_data": topic_data,
                    "source_meta": source_meta
                }
    except Exception:
        logger.exception("Failed to fetch last live cache.")
        return None


# -------------------------
# Entrypoints & table setup
# -------------------------
def create_tables() -> None:
    """
    Create synthetic tables (destructive) and ensure live cache tables exist (non-destructive).
    NOTE: This keeps previous live cache data and adds live tables if missing.
    """
    logger.info("Stage 3.A: Creating/recreating core synthetic tables and ensuring live cache tables exist.")
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                # destructive recreate of core synthetic tables
                cur.execute(CREATE_TABLES_SQL)
                # ensure live tables exist (non-destructive)
                cur.execute(LIVE_CACHE_SQL)
                conn.commit()
        logger.info("Tables created/ensured successfully.")
    except Exception:
        logger.exception("Failed to create/ensure tables.")


def run_etl() -> bool:
    """
    Truncate synthetic tables then run chosen ETL (Pandas or Spark).
    Live ingestion is not automatically run here; use ingest_live_cache for live data.
    """
    logger.info("Stage 3.B: Running ETL pipeline")
    mode = "Spark" if config.USE_SPARK_ETL else "Pandas"
    logger.info("ETL mode = %s", mode)
    
    ok, msg = utils.check_file_prerequisites(3)
    if not ok:
        logger.error("Missing prerequisites for ETL: %s", msg)
        return False
    
    # Truncate core tables for a fresh ingest
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(TRUNCATE_STAGE3_SQL)
                conn.commit()
        logger.info("Truncated synthetic tables.")
    except Exception:
        logger.exception("Failed to truncate core tables; proceeding anyway (in case tables are fresh).")
    
    if config.USE_SPARK_ETL:
        return _run_etl_spark()
    else:
        return _run_etl_pandas()


# -------------------------
# If executed as script
# -------------------------
if __name__ == "__main__":
    create_tables()
    run_etl()
