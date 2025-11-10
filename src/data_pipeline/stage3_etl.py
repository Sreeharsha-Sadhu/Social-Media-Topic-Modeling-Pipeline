"""
stage_3_etl.py
────────────────────────────────────────────
Stage 3: Extract–Transform–Load (ETL)

Supports dual modes (Pandas / PySpark) controlled via .env:
    USE_SPARK_ETL=true
    SPARK_MASTER=local[*]
    SPARK_APP_NAME=SocialMediaETL

Purpose:
    - Load synthetic user, post, and follow data from JSON.
    - Clean and normalize text + timestamps.
    - Write data into PostgreSQL via JDBC (Spark) or SQLAlchemy (Pandas).

Outputs:
    Database tables:
        - users
        - follows
        - posts
"""

import pandas as pd
from sqlalchemy.types import ARRAY, String
from pyspark.sql import functions as F
from tqdm import tqdm
from typing import Optional

from src.core import config, utils
from src.core.logging_config import get_logger

logger = get_logger(__name__)

# ───────────────────────────────────────────────────────────────
# SQL Definitions
# ───────────────────────────────────────────────────────────────
CREATE_TABLES_SQL = """
DROP TABLE IF EXISTS posts CASCADE;
DROP TABLE IF EXISTS follows CASCADE;
DROP TABLE IF EXISTS users CASCADE;

CREATE TABLE IF NOT EXISTS users (
    user_id    TEXT PRIMARY KEY,
    username   TEXT,
    personas   TEXT[],
    subreddits TEXT[]
);

CREATE TABLE IF NOT EXISTS follows (
    follower_id TEXT REFERENCES users (user_id) ON DELETE CASCADE,
    followed_id TEXT REFERENCES users (user_id) ON DELETE CASCADE,
    PRIMARY KEY (follower_id, followed_id)
);

CREATE TABLE IF NOT EXISTS posts (
    post_id         TEXT PRIMARY KEY,
    author_id       TEXT REFERENCES users (user_id) ON DELETE CASCADE,
    content         TEXT,
    cleaned_content TEXT,
    created_at      TIMESTAMP
);
"""

TRUNCATE_STAGE3_SQL = "TRUNCATE TABLE users, follows, posts RESTART IDENTITY CASCADE;"


# ───────────────────────────────────────────────────────────────
# Spark Initialization
# ───────────────────────────────────────────────────────────────
def _init_spark():
    """Initialize or return an existing SparkSession."""
    try:
        if hasattr(utils, "get_spark_session"):
            return utils.get_spark_session()
    except Exception:
        pass

    from pyspark.sql import SparkSession
    logger.info(f"🧩 Initializing SparkSession for {config.SPARK_APP_NAME} ...")
    spark = (
        SparkSession.builder
        .appName(config.SPARK_APP_NAME)
        .master(config.SPARK_MASTER)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        .config("spark.sql.repl.eagerEval.enabled", "true")
        .getOrCreate()
    )
    logger.info("✅ SparkSession started.")
    return spark


# ───────────────────────────────────────────────────────────────
# Pandas ETL Path
# ───────────────────────────────────────────────────────────────
def _run_etl_pandas() -> bool:
    """Run ETL using Pandas with tqdm progress tracking."""
    logger.info("🐼 Running Pandas ETL ...")

    try:
        engine = utils.get_sqlalchemy_engine()

        # USERS
        df_users = pd.read_json(config.USERS_JSON_PATH, orient="records")
        logger.info(f"Inserting {len(df_users)} users ...")
        df_users.to_sql(
            "users",
            engine,
            if_exists="append",
            index=False,
            dtype={"personas": ARRAY(String), "subreddits": ARRAY(String)},
        )

        # FOLLOWS
        df_follows = pd.read_json(config.FOLLOWS_JSON_PATH, orient="records")
        logger.info(f"Inserting {len(df_follows)} follows ...")
        df_follows.to_sql("follows", engine, if_exists="append", index=False)

        # POSTS
        df_posts = pd.read_json(config.POSTS_JSON_PATH, orient="records")
        logger.info(f"Cleaning {len(df_posts)} posts ...")

        for i in tqdm(range(len(df_posts)), desc="Cleaning posts", unit="post"):
            pass  # visual progress only (not CPU-bound)

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

        logger.info("Writing cleaned posts to PostgreSQL ...")
        df_posts.to_sql("posts", engine, if_exists="append", index=False)

        logger.info("✅ Pandas ETL completed successfully.")
        return True

    except Exception as e:
        logger.error(f"🚨 Pandas ETL failed: {e}")
        return False

    finally:
        if "engine" in locals():
            engine.dispose()
        logger.info("ETL process finished (Pandas).")


# ───────────────────────────────────────────────────────────────
# Spark ETL Path
# ───────────────────────────────────────────────────────────────
def _run_etl_spark() -> bool:
    """Run ETL using Spark; fully vectorized, no UDFs."""
    logger.info("🚀 Running Spark ETL ...")

    try:
        spark = _init_spark()
        jdbc_url = f"jdbc:postgresql://{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
        jdbc_opts = {
            "url": jdbc_url,
            "user": config.DB_USER,
            "password": config.DB_PASS,
            "driver": "org.postgresql.Driver",
        }

        # USERS
        logger.info("Loading users.json ...")
        df_users = spark.read.option("multiLine", "true").json(str(config.USERS_JSON_PATH))
        df_users.write.format("jdbc").options(**jdbc_opts, dbtable="users").mode("append").save()
        logger.info(f"Inserted {df_users.count()} users.")

        # FOLLOWS
        logger.info("Loading follows.json ...")
        df_follows = spark.read.option("multiLine", "true").json(str(config.FOLLOWS_JSON_PATH))
        df_follows.write.format("jdbc").options(**jdbc_opts, dbtable="follows").mode("append").save()
        logger.info(f"Inserted {df_follows.count()} follows.")

        # POSTS
        logger.info("Loading posts.json and cleaning content ...")
        df_posts = spark.read.option("multiLine", "true").json(str(config.POSTS_JSON_PATH))

        total_posts = df_posts.count()
        logger.info(f"Cleaning {total_posts} posts ...")
        for _ in tqdm(range(total_posts), desc="Processing posts", unit="post"):
            pass  # lightweight progress indicator

        df_posts = df_posts.withColumn("cleaned_content", F.lower(F.col("content")))
        df_posts = df_posts.withColumn("cleaned_content", F.regexp_replace("cleaned_content", r"http\\S+", ""))
        df_posts = df_posts.withColumn("cleaned_content", F.regexp_replace("cleaned_content", r"[^a-z0-9\\s]", ""))
        df_posts = df_posts.withColumn("cleaned_content", F.regexp_replace("cleaned_content", r"\\s+", " "))
        df_posts = df_posts.withColumn("cleaned_content", F.trim(F.col("cleaned_content")))

        # Safe timestamp normalization
        df_posts = df_posts.withColumn("created_at", F.regexp_replace("created_at", "Z$", ""))
        df_posts = df_posts.withColumn(
            "created_at",
            F.expr("try_to_timestamp(created_at, \"yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX\")"),
        )

        logger.info("Writing posts to PostgreSQL via JDBC ...")
        df_posts.write.format("jdbc").options(**jdbc_opts, dbtable="posts").mode("append").save()

        logger.info("✅ Spark ETL completed successfully.")
        return True

    except Exception as e:
        logger.error(f"🚨 Spark ETL Error: {e}")
        if "CAST_INVALID_INPUT" in str(e) or "TIMESTAMP" in str(e):
            logger.warning("Timestamp format issue detected. Falling back to Pandas ETL ...")
            return _run_etl_pandas()
        return False

    finally:
        try:
            if not hasattr(utils, "get_spark_session"):
                spark.stop()
        except Exception:
            pass
        logger.info("ETL process finished (Spark).")


# ───────────────────────────────────────────────────────────────
# Entrypoints
# ───────────────────────────────────────────────────────────────
def create_tables() -> None:
    """Drop + recreate Stage 3 tables."""
    logger.info("🧱 Stage 3.A: Create Core Tables")
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLES_SQL)
                conn.commit()
        logger.info("✅ Tables 'users', 'follows', and 'posts' created successfully.")
    except Exception as e:
        logger.error(f"🚨 Error creating tables: {e}")


def run_etl() -> bool:
    """Truncate existing data and run selected ETL mode."""
    logger.info("📦 Stage 3.B: Run ETL")
    mode = "Spark" if config.USE_SPARK_ETL else "Pandas"
    logger.info(f"ETL Mode → {mode}")

    ok, msg = utils.check_file_prerequisites(3)
    if not ok:
        logger.error(f"🚨 Missing prerequisites: {msg}")
        return False

    # Truncate old data
    logger.info("Truncating Stage 3 tables ...")
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(TRUNCATE_STAGE3_SQL)
                conn.commit()
        logger.info("✅ Tables truncated successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Could not truncate tables: {e}")

    # Execute ETL path
    if config.USE_SPARK_ETL:
        return _run_etl_spark()
    else:
        return _run_etl_pandas()


# ───────────────────────────────────────────────────────────────
# Script Entrypoint
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    create_tables()
    run_etl()
