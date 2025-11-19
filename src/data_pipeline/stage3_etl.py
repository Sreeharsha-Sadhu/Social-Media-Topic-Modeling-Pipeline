"""
stage_3_etl.py (unified)
──────────────────────────────────────────────────────
Creates DB schema for Reddit-first application.
This ETL no longer ingests synthetic posts. Posts are populated
by the live fetcher (PRAW primary, simulator fallback).

Key functions:
 - create_tables() : recreate users/follows/posts schema (destructive for core synthetic tables)
 - run_etl()       : currently creates tables and optionally seeds users (does NOT insert posts)
 - ingest_live_cache/get_last_live_cache/record_live_run : live persistence helpers (unchanged)
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

# -------------------------
# SQL — core synthetic tables (users + follows) and full Reddit posts schema
# -------------------------
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
        subreddit       TEXT,
        title           TEXT,
        selftext        TEXT,
        content         TEXT,
        cleaned_content TEXT,
        created_at      TIMESTAMPTZ,
        score           INTEGER,
        num_comments    INTEGER,
        flair           TEXT
    );
"""

TRUNCATE_STAGE3_SQL = "TRUNCATE TABLE users, follows RESTART IDENTITY CASCADE;"


# Live tables (ensure exists)
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
        ON live_feed_cache (user_id, platform, fetched_at DESC);
"""


# -------------------------
# Spark session init (unchanged)
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
# NOTE: THIS ETL NO LONGER IMPORTS POSTS JSON — posts should come from live fetcher.
# -------------------------
def _run_etl_pandas() -> bool:
    """
    Previously this inserted synthetic posts; now we only insert users and follows
    from provided JSON if present. Posts table is left empty for live ingestion.
    """
    logger.info("Running Pandas ETL (users & follows only) ...")
    try:
        engine = utils.get_sqlalchemy_engine()

        if getattr(config, "USERS_JSON_PATH", None):
            df_users = pd.read_json(config.USERS_JSON_PATH, orient="records")
            logger.info("Inserting users (from JSON)...")
            df_users.to_sql("users", engine, if_exists="append", index=False,
                            dtype={"personas": ARRAY(String), "subreddits": ARRAY(String)})
        else:
            logger.info("No USERS_JSON_PATH provided; skipping user ingestion.")

        if getattr(config, "FOLLOWS_JSON_PATH", None):
            df_follows = pd.read_json(config.FOLLOWS_JSON_PATH, orient="records")
            logger.info("Inserting follows (from JSON)...")
            df_follows.to_sql("follows", engine, if_exists="append", index=False)
        else:
            logger.info("No FOLLOWS_JSON_PATH provided; skipping follows ingestion.")

        logger.info("Pandas ETL (users & follows) completed.")
        return True
    except Exception as e:
        logger.exception("Pandas ETL failed.")
        return False
    finally:
        if "engine" in locals():
            engine.dispose()


def _run_etl_spark() -> bool:
    """
    Spark variant — only imports users & follows if JSON paths are provided.
    """
    logger.info("Running Spark ETL (users & follows only) ...")
    try:
        spark = _init_spark()
        jdbc_url = f"jdbc:postgresql://{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
        jdbc_opts = {
            "url": jdbc_url,
            "user": config.DB_USER,
            "password": config.DB_PASS,
            "driver": "org.postgresql.Driver",
        }

        if getattr(config, "USERS_JSON_PATH", None):
            logger.info("Loading users.json into Spark...")
            df_users = spark.read.option("multiLine", "true").json(str(config.USERS_JSON_PATH))
            df_users.write.format("jdbc").options(**jdbc_opts, dbtable="users").mode("append").save()
            logger.info("Users inserted via JDBC.")
        else:
            logger.info("No USERS_JSON_PATH provided; skipping users.")

        if getattr(config, "FOLLOWS_JSON_PATH", None):
            df_follows = spark.read.option("multiLine", "true").json(str(config.FOLLOWS_JSON_PATH))
            df_follows.write.format("jdbc").options(**jdbc_opts, dbtable="follows").mode("append").save()
            logger.info("Follows inserted via JDBC.")
        else:
            logger.info("No FOLLOWS_JSON_PATH provided; skipping follows.")

        logger.info("Spark ETL (users & follows) completed.")
        return True
    except Exception as e:
        logger.exception("Spark ETL failed.")
        return False
    finally:
        try:
            if not hasattr(utils, "get_spark_session"):
                spark.stop()
        except Exception:
            pass


# -------------------------
# Live ingestion helpers (unchanged behaviour)
# -------------------------
def record_live_run(user_id: Optional[str], platform: str, run_status: str, notes: Optional[str] = None) -> None:
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
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO live_feed_cache (user_id, platform, raw_posts, summary, topic_data, source_meta)
                    VALUES (%s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb)
                    RETURNING cache_id;
                    """,
                    (user_id,
                     platform,
                     json.dumps(raw_posts) if raw_posts is not None else None,
                     summary_text,
                     json.dumps(topic_rows) if topic_rows is not None else None,
                     json.dumps(source_meta) if source_meta is not None else None)
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
    Posts are created with the full Reddit schema.
    """
    logger.info("Stage 3.A: Creating/recreating core synthetic tables and ensuring live cache tables exist.")
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLES_SQL)
                cur.execute(LIVE_CACHE_SQL)
                conn.commit()
        logger.info("Tables created/ensured successfully.")
    except Exception:
        logger.exception("Failed to create/ensure tables.")


def run_etl(seed_users: bool = False) -> bool:
    """
    Run ETL: only recreates/truncates user/follows tables then runs user ingestion (optional).
    Posts are intentionally not inserted here — they will be populated by live fetches.
    """
    logger.info("Stage 3.B: Running ETL pipeline (users/follows only)")
    ok, msg = utils.check_file_prerequisites(3)
    if not ok:
        logger.error("Missing prerequisites for ETL: %s", msg)
        # proceed if user explicitly asked to seed users
        if not seed_users:
            return False

    # Truncate user/follows for a fresh ingest
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(TRUNCATE_STAGE3_SQL)
                conn.commit()
        logger.info("Truncated user/follows tables.")
    except Exception:
        logger.exception("Failed to truncate core tables; proceeding anyway.")

    if config.USE_SPARK_ETL:
        return _run_etl_spark()
    else:
        return _run_etl_pandas()


if __name__ == "__main__":
    create_tables()
    run_etl()
