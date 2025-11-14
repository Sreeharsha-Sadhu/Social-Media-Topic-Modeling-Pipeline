"""
utils.py
--------------------------------------------------------
Core utility helpers for the MMDS Project.

Responsibilities:
- Database connections (psycopg2 and SQLAlchemy)
- Spark session initialization (singleton)
- File prerequisite checks
- Live-analysis DB state helpers
- Console clearing
- Logging integration

This file is imported by nearly every stage in the pipeline.
"""

import os
import sys
from typing import Tuple, Optional, Dict, Any

import psycopg2
from pyspark import SparkConf
from pyspark.sql import SparkSession
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.core.config import (
    DB_NAME, DB_USER, DB_PASS, DB_HOST, DB_PORT,
    USERS_JSON_PATH, POSTS_JSON_PATH, FOLLOWS_JSON_PATH,
)
from src.core.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

# Spark singleton
_SPARK_SESSION: Optional[SparkSession] = None


# ================================================================
# Database Helpers
# ================================================================
def get_db_connection():
    """
    Returns a fresh psycopg2 database connection.
    """
    try:
        conn_str = (
            f"dbname='{DB_NAME}' user='{DB_USER}' "
            f"password='{DB_PASS}' host='{DB_HOST}' port='{DB_PORT}'"
        )
        conn = psycopg2.connect(conn_str)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


def get_sqlalchemy_engine() -> Engine:
    """
    Returns a SQLAlchemy engine for Pandas integration (to_sql/read_sql).
    """
    try:
        conn_str = (
            f"postgresql+psycopg2://{DB_USER}:{DB_PASS}"
            f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        engine = create_engine(conn_str)
        return engine
    except Exception as e:
        logger.error(f"SQLAlchemy engine creation failed: {e}")
        raise


# ================================================================
# Spark Session (Singleton)
# ================================================================
def get_spark_session() -> SparkSession:
    """
    Returns a global SparkSession instance with stable Python worker settings.
    Ensures:
    - Correct Python executable is used across workers
    - Arrow optimizations enabled where safe
    - Worker reuse
    """
    global _SPARK_SESSION
    
    if _SPARK_SESSION is not None:
        return _SPARK_SESSION
    
    logger.info("Initializing SparkSession...")
    
    try:
        conf = SparkConf()
        conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        conf.set("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        conf.set("spark.python.worker.reuse", "true")
        conf.set("spark.default.parallelism", "4")
        conf.set("spark.executorEnv.PYSPARK_PYTHON", sys.executable)
        conf.set("spark.executorEnv.PYSPARK_DRIVER_PYTHON", sys.executable)
        
        _SPARK_SESSION = (
            SparkSession.builder
            .appName("MMDS Spark Engine")
            .master("local[*]")
            .config(conf=conf)
            .getOrCreate()
        )
        
        _SPARK_SESSION.sparkContext.setLogLevel("WARN")
        logger.info("SparkSession initialized successfully.")
    
    except Exception as e:
        logger.error(f"Failed to initialize SparkSession: {e}")
        raise
    
    return _SPARK_SESSION


# ================================================================
# File Prerequisite Checks (Used by CLI Stages)
# ================================================================
def check_file_prerequisites(stage_number: int) -> Tuple[bool, Optional[str]]:
    """
    Returns (True, None) if files required for the stage exist.
    Otherwise returns (False, error_message).
    """
    if stage_number == 1:
        return True, None
    
    if stage_number == 2:
        if not USERS_JSON_PATH.exists():
            return False, f"Missing file: {USERS_JSON_PATH}. Run Stage 1 first."
        return True, None
    
    if stage_number == 3:
        missing = []
        if not USERS_JSON_PATH.exists():
            missing.append(str(USERS_JSON_PATH))
        if not POSTS_JSON_PATH.exists():
            missing.append(str(POSTS_JSON_PATH))
        if not FOLLOWS_JSON_PATH.exists():
            missing.append(str(FOLLOWS_JSON_PATH))
        
        if missing:
            return False, f"Missing files: {', '.join(missing)}. Run Stages 1 and 2."
    
    return True, None


# ================================================================
# Console Helper
# ================================================================
def clear_screen():
    """
    Clears console output.
    """
    os.system("cls" if os.name == "nt" else "clear")


# ================================================================
# --- Stateful Live Analysis Helpers ---
# ================================================================
def save_live_analysis(
        user_id: str,
        platform: str,
        payload: Dict[str, Any],
) -> None:
    """
    Store the latest live-analysis results (Reddit, LinkedIn, etc.)
    into database in a stateful manner. If the platform entry exists
    for the user, overwrite it.
    """
    logger.info(f"Saving live analysis for user={user_id}, platform={platform}")
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                            CREATE TABLE IF NOT EXISTS live_analysis_state
                            (
                                user_id    TEXT,
                                platform   TEXT,
                                data       JSONB,
                                updated_at TIMESTAMP DEFAULT NOW(),
                                PRIMARY KEY (user_id, platform)
                            );
                            """)
                
                cur.execute("""
                            INSERT INTO live_analysis_state (user_id, platform, data)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (user_id, platform)
                                DO UPDATE SET data       = EXCLUDED.data,
                                              updated_at = NOW();
                            """, (user_id, platform, psycopg2.extras.Json(payload)))
                
                conn.commit()
    
    except Exception as e:
        logger.error(f"Failed to store live analysis output: {e}")
        raise


def load_live_analysis(
        user_id: str,
        platform: str
) -> Optional[Dict[str, Any]]:
    """
    Load previously saved live-analysis state.
    Returns None if no entry exists.
    """
    logger.info(f"Loading saved live analysis for user={user_id}, platform={platform}")
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                            SELECT data
                            FROM live_analysis_state
                            WHERE user_id = %s
                              AND platform = %s;
                            """, (user_id, platform))
                
                row = cur.fetchone()
                return row[0] if row else None
    
    except Exception as e:
        logger.error(f"Failed to load live analysis state: {e}")
        return None


def fetch_last_live_summary(user_id, source):
    engine = get_sqlalchemy_engine()
    query = """
            SELECT *
            FROM live_topics
            WHERE user_id = %s
              AND source = %s
            ORDER BY created_at DESC
            LIMIT 1; \
            """
    df = pd.read_sql(query, engine, params=[user_id, source])
    engine.dispose()
    return df if not df.empty else None
