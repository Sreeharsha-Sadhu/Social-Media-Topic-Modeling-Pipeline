"""
Stage 3: ETL (Dual-mode: Pandas or Spark)
────────────────────────────────────────────
Runs the Extract-Transform-Load stage using either Pandas or PySpark,
depending on the USE_SPARK_ETL flag in .env / settings.py

Environment flags:
    USE_SPARK_ETL=true
    SPARK_MASTER=local[*]
    SPARK_APP_NAME=SocialMediaETL
"""

import pandas as pd
import psycopg2
from sqlalchemy.types import ARRAY, String
from pyspark.sql import functions as F

from src.common import utils
from src.config import settings


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

TRUNCATE_STAGE3_SQL = """
TRUNCATE TABLE users, follows, posts RESTART IDENTITY CASCADE;
"""


# ───────────────────────────────────────────────────────────────
# Spark initialization
# ───────────────────────────────────────────────────────────────
def _init_spark():
    """Return an active SparkSession (prefers utils.get_spark_session)."""
    try:
        if hasattr(utils, "get_spark_session"):
            return utils.get_spark_session()
    except Exception:
        pass

    from pyspark.sql import SparkSession

    print(f"🧩 Initializing SparkSession for {settings.SPARK_APP_NAME} ...")
    spark = (
        SparkSession.builder
        .appName(getattr(settings, "SPARK_APP_NAME", "SocialMediaETL"))
        .master(getattr(settings, "SPARK_MASTER", "local[*]"))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        .config("spark.sql.repl.eagerEval.enabled", "true")
        .getOrCreate()
    )
    print("✅ SparkSession started")
    return spark


# ───────────────────────────────────────────────────────────────
# Pandas ETL path
# ───────────────────────────────────────────────────────────────
def _run_etl_pandas():
    """Run ETL using Pandas only."""
    print("--- 🐼 Running Pandas ETL ---")
    try:
        engine = utils.get_sqlalchemy_engine()

        # USERS
        print("Loading users.json ...")
        df_users = pd.read_json(settings.USERS_JSON_PATH, orient="records")
        df_users.to_sql(
            "users",
            engine,
            if_exists="append",
            index=False,
            dtype={"personas": ARRAY(String), "subreddits": ARRAY(String)},
        )
        print(f"Inserted {len(df_users)} users.")

        # FOLLOWS
        print("Loading follows.json ...")
        df_follows = pd.read_json(settings.FOLLOWS_JSON_PATH, orient="records")
        df_follows.to_sql("follows", engine, if_exists="append", index=False)
        print(f"Inserted {len(df_follows)} follows.")

        # POSTS
        print("Loading posts.json and cleaning content ...")
        df_posts = pd.read_json(settings.POSTS_JSON_PATH, orient="records")

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

        df_posts.to_sql("posts", engine, if_exists="append", index=False)
        print(f"Inserted {len(df_posts)} posts.")

        print("\n✅ Pandas ETL completed successfully.")
        return True

    except Exception as e:
        print("\n🚨 ERROR during Pandas ETL")
        print(f"Details: {e}")
        return False

    finally:
        if "engine" in locals():
            engine.dispose()
        print("ETL process finished (Pandas).")


# ───────────────────────────────────────────────────────────────
# Spark ETL path
# ───────────────────────────────────────────────────────────────
def _run_etl_spark():
    """Run ETL using Spark; safe, UDF-free, with robust timestamp parsing."""
    print("--- 🚀 Running Spark ETL ---")
    try:
        spark = _init_spark()
        jdbc_url = f"jdbc:postgresql://{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        jdbc_opts = {
            "url": jdbc_url,
            "user": settings.DB_USER,
            "password": settings.DB_PASS,
            "driver": "org.postgresql.Driver",
        }

        # USERS
        print("Reading users.json via Spark...")
        df_users = spark.read.option("multiLine", "true").json(settings.USERS_JSON_PATH)
        print("Writing 'users' table ...")
        (
            df_users.write.format("jdbc")
            .options(**jdbc_opts, dbtable="users")
            .mode("append")
            .save()
        )

        # FOLLOWS
        print("Reading follows.json via Spark...")
        df_follows = spark.read.option("multiLine", "true").json(settings.FOLLOWS_JSON_PATH)
        print("Writing 'follows' table ...")
        (
            df_follows.write.format("jdbc")
            .options(**jdbc_opts, dbtable="follows")
            .mode("append")
            .save()
        )

        # POSTS
        print("Reading posts.json via Spark...")
        df_posts = spark.read.option("multiLine", "true").json(settings.POSTS_JSON_PATH)

        print("Cleaning post content (Spark native)...")
        df_posts = df_posts.withColumn("cleaned_content", F.lower(F.col("content")))
        df_posts = df_posts.withColumn("cleaned_content", F.regexp_replace("cleaned_content", r"http\\S+", ""))
        df_posts = df_posts.withColumn("cleaned_content", F.regexp_replace("cleaned_content", r"[^a-z0-9\\s]", ""))
        df_posts = df_posts.withColumn("cleaned_content", F.regexp_replace("cleaned_content", r"\\s+", " "))
        df_posts = df_posts.withColumn("cleaned_content", F.trim(F.col("cleaned_content")))

        # Safe timestamp normalization
        print("Normalizing timestamps ...")
        df_posts = df_posts.withColumn("created_at", F.regexp_replace("created_at", "Z$", ""))
        df_posts = df_posts.withColumn(
            "created_at",
            F.expr("try_to_timestamp(created_at, \"yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX\")"),
        )

        print("Writing 'posts' table ...")
        (
            df_posts.write.format("jdbc")
            .options(**jdbc_opts, dbtable="posts")
            .mode("append")
            .save()
        )

        print("\n✅ Spark ETL completed successfully.")
        return True

    except Exception as e:
        print(f"\n🚨 Spark ETL Error: {e}")
        if "CAST_INVALID_INPUT" in str(e) or "TIMESTAMP" in str(e):
            print("Falling back to Pandas ETL (timestamp parse issue).")
            return _run_etl_pandas()
        else:
            print("Unhandled Spark error. Manual inspection required.")
            return False

    finally:
        try:
            if not hasattr(utils, "get_spark_session"):
                spark.stop()
        except Exception:
            pass
        print("ETL process finished (Spark).")


# ───────────────────────────────────────────────────────────────
# Entrypoints
# ───────────────────────────────────────────────────────────────
def create_tables():
    """Drop + recreate Stage 3 tables."""
    print("--- Stage 3.A: Create Core Tables ---")
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLES_SQL)
                conn.commit()
        print("✅ Tables 'users', 'follows', 'posts' created successfully.")
    except Exception as e:
        print(f"🚨 Error creating tables: {e}")


def run_etl():
    """Truncate existing data and run the selected ETL mode."""
    print("\n--- Stage 3.B: Run ETL ---")
    mode = "Spark" if getattr(settings, "USE_SPARK_ETL", False) else "Pandas"
    print(f"ETL Mode: {mode}")

    # prerequisite check
    ok, msg = utils.check_file_prerequisites(3)
    if not ok:
        print(f"🚨 {msg}")
        return False

    # Truncate tables (always clean start)
    print("Truncating existing data ...")
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(TRUNCATE_STAGE3_SQL)
                conn.commit()
        print("✅ Tables truncated.")
    except Exception as e:
        print(f"⚠️ Warning: could not truncate tables: {e}")

    # Execute chosen ETL path
    return _run_etl_spark() if getattr(settings, "USE_SPARK_ETL", False) else _run_etl_pandas()


# ───────────────────────────────────────────────────────────────
# Script entrypoint
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    create_tables()
    run_etl()
