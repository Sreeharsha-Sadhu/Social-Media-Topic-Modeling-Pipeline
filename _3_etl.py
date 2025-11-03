# stage_3_etl.py

import psycopg2
from psycopg2 import sql
import re
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
import config
import utils

# SQL to create the base tables
CREATE_TABLES_SQL = """
                    CREATE TABLE IF NOT EXISTS users \
                    ( \
                        user_id  TEXT PRIMARY KEY, \
                        username TEXT, \
                        personas TEXT[]
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

# --- MODIFIED SECTION ---
# This is now a single, robust SQL command.
# It attempts to truncate all 5 tables. The RESTART IDENTITY CASCADE
# handles all foreign key relationships automatically.
TRUNCATE_ALL_SQL = """
                   TRUNCATE TABLE
                       users,
                       follows,
                       posts,
                       user_topics,
                       post_topic_mapping
                       RESTART IDENTITY CASCADE; \
                   """

# This is the fallback SQL if the Stage 4 tables don't exist yet.
TRUNCATE_STAGE3_SQL = """
                      TRUNCATE TABLE
                          users,
                          follows,
                          posts
                          RESTART IDENTITY CASCADE; \
                      """


# --- END MODIFIED SECTION ---


def create_tables():
    """Creates the core DB tables if they don't exist."""
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
    """Runs the full ETL pipeline: Truncate and load with PySpark."""
    print("\n--- Starting Stage 3.B: PySpark ETL (SQL-Native Functions Only) ---")
    
    # --- 1. Prerequisite Check ---
    success, msg = utils.check_file_prerequisites(3)
    if not success:
        print(f"🚨 ERROR: {msg}")
        return False
    
    # --- 2. Truncate Tables (NEW ROBUST LOGIC) ---
    print("Stage 3.B.1: Truncating old data...")
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                try:
                    # First, try to truncate ALL 5 tables.
                    cur.execute(TRUNCATE_ALL_SQL)
                except psycopg2.errors.UndefinedTable:
                    # If Stage 4 tables don't exist, this fails.
                    # Roll back the failed transaction
                    conn.rollback()
                    print("  (Note: Stage 4 tables not found. Truncating Stage 3 tables only.)")
                    # Execute the fallback truncate for just Stage 3 tables
                    cur.execute(TRUNCATE_STAGE3_SQL)
                
                # Commit the successful transaction
                conn.commit()
        print("Successfully truncated tables.")
    except Exception as e:
        print(f"--- 🚨 ERROR: Could not truncate tables ---")
        print(f"Error details: {e}")
        return False
    # --- END NEW TRUNCATE LOGIC ---
    
    # --- 3. Run PySpark ETL ---
    print("Stage 3.B.2: Initializing SparkSession...")
    spark = (
        SparkSession.builder
        .appName("SocialMediaETL")
        .master("local[*]")
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3")
        .getOrCreate()
    )
    print("SparkSession initialized.")
    
    # JDBC Config
    jdbc_url = f"jdbc:postgresql://{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
    properties = {"user": config.DB_USER, "password": config.DB_PASS, "driver": "org.postgresql.Driver"}
    
    try:
        # --- USERS ---
        print("Processing 'users' data...")
        users_df = spark.read.option("multiline", "true").json(config.USERS_JSON_PATH)
        users_df.write.jdbc(jdbc_url, "users", mode="append", properties=properties)
        print(f"Successfully loaded {users_df.count()} users.")
        
        # --- FOLLOWS ---
        print("Processing 'follows' data...")
        follows_df = spark.read.option("multiline", "true").json(config.FOLLOWS_JSON_PATH)
        follows_df.write.jdbc(jdbc_url, "follows", mode="append", properties=properties)
        print(f"Successfully loaded {follows_df.count()} follows.")
        
        # --- POSTS ---
        print("Processing 'posts' data (SQL-native functions)...")
        posts_df = spark.read.option("multiline", "true").json(config.POSTS_JSON_PATH)
        
        # SQL-native cleaning logic
        cleaned_col = F.lower(F.col("content"))
        cleaned_col = F.regexp_replace(cleaned_col, "http\\S+", "")
        cleaned_col = F.regexp_replace(cleaned_col, "[^a-z0-9\\s]", "")
        cleaned_col = F.regexp_replace(cleaned_col, "\\s+", " ")
        cleaned_col = F.trim(cleaned_col)
        cleaned_col = F.when(cleaned_col == "", None).otherwise(cleaned_col)
        
        posts_df_final = posts_df.withColumn("cleaned_content", cleaned_col) \
            .withColumn("created_at", F.to_timestamp(F.col("created_at"))) \
            .select("post_id", "author_id", "content", "cleaned_content", "created_at")
        
        posts_df_final.write.jdbc(jdbc_url, "posts", mode="append", properties=properties)
        print(f"Successfully loaded {posts_df_final.count()} posts.")
        
        print("\n--- PySpark ETL Job Complete! ---")
        return True
    
    except Exception as e:
        print(f"\n--- 🚨 ERROR during Spark ETL ---")
        print(f"Error details: {e}")
        return False
    
    finally:
        spark.stop()
        print("SparkSession stopped.")


if __name__ == "__main__":
    # Allows running: python stage_3_etl.py
    create_tables()
    run_etl()
