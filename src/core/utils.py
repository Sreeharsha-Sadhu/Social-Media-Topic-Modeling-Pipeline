"""
utils.py
----------------------------------
Shared utility functions for database connections, Spark initialization,
and file operations within the MMDS Project.

This module is designed for safe re-use across both Pandas and Spark
pipelines. All logs are routed through the unified logging system.
"""

import os
import sys
import psycopg2
import logging
from typing import Tuple, Optional

from sqlalchemy import create_engine
from pyspark.sql import SparkSession
from pyspark import SparkConf
from IPython.display import clear_output

from src.core import config
from src.core.logging_config import get_logger

# Initialize module-level logger
logger = get_logger(__name__)

# Singleton SparkSession instance
_spark_session: Optional[SparkSession] = None


# ---------------------------------------------------------------------
# Database Utilities
# ---------------------------------------------------------------------
def get_db_connection() -> psycopg2.extensions.connection:
    """
    Establishes a new psycopg2 connection to the configured PostgreSQL database.

    Returns:
        psycopg2.extensions.connection: Active database connection.

    Raises:
        psycopg2.OperationalError: If connection fails.
    """
    conn_string = (
        f"dbname='{config.DB_NAME}' "
        f"user='{config.DB_USER}' "
        f"password='{config.DB_PASS}' "
        f"host='{config.DB_HOST}' "
        f"port='{config.DB_PORT}'"
    )
    try:
        conn = psycopg2.connect(conn_string)
        logger.info("✅ Connected to PostgreSQL database successfully.")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"🚨 Database connection failed: {e}")
        raise


def get_sqlalchemy_engine():
    """
    Creates a new SQLAlchemy engine for pandas.to_sql() and general ORM use.

    Returns:
        sqlalchemy.Engine: Configured SQLAlchemy engine.
    """
    conn_string = (
        f"postgresql+psycopg2://{config.DB_USER}:{config.DB_PASS}"
        f"@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
    )
    logger.info("Creating SQLAlchemy engine...")
    return create_engine(conn_string, echo=False)


# ---------------------------------------------------------------------
# Spark Utilities
# ---------------------------------------------------------------------
def get_spark_session() -> SparkSession:
    """
    Returns a singleton SparkSession configured for the MMDS Project.

    Ensures:
    - Consistent Python executable across driver and workers.
    - Reasonable default parallelism and memory settings.
    - Arrow-based DataFrame optimization.
    """
    global _spark_session

    if _spark_session is not None:
        logger.debug("Using existing SparkSession instance.")
        return _spark_session

    logger.info(f"🧩 Initializing SparkSession: {config.SPARK_APP_NAME} [{config.SPARK_MASTER}]")

    conf = SparkConf()

    # Core configuration
    conf.setAppName(config.SPARK_APP_NAME)
    conf.setMaster(config.SPARK_MASTER)

    # Stability and performance tuning
    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    conf.set("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
    conf.set("spark.python.worker.reuse", "true")
    conf.set("spark.default.parallelism", "4")
    conf.set("spark.sql.shuffle.partitions", "4")
    conf.set("spark.executor.memory", "3g")
    conf.set("spark.driver.memory", "6g")
    conf.set("spark.python.worker.memory", "512m")
    conf.set("spark.python.profile", "true")

    # Match Python executables across environments
    python_exe = sys.executable
    conf.set("spark.executorEnv.PYSPARK_PYTHON", python_exe)
    conf.set("spark.executorEnv.PYSPARK_DRIVER_PYTHON", python_exe)

    try:
        _spark_session = SparkSession.builder.config(conf=conf).getOrCreate()
        _spark_session.sparkContext.setLogLevel("WARN")
        logger.info("✅ SparkSession started successfully.")
    except Exception as e:
        logger.error(f"🚨 Failed to initialize SparkSession: {e}")
        raise

    return _spark_session


# ---------------------------------------------------------------------
# Console Utilities
# ---------------------------------------------------------------------
def clear_screen() -> None:
    """
    Clears the console screen across platforms.
    Useful for CLI applications or notebook outputs.
    """
    try:
        clear_output(wait=True)
    except Exception:
        os.system("cls" if os.name == "nt" else "clear")


# ---------------------------------------------------------------------
# Stage File Checks
# ---------------------------------------------------------------------
def check_file_prerequisites(stage_number: int) -> Tuple[bool, Optional[str]]:
    """
    Checks if required files exist for a specific processing stage.

    Args:
        stage_number (int): The stage ID (1–5).

    Returns:
        (bool, Optional[str]): Tuple of (status, error_message).
    """
    required_files = []

    if stage_number == 1:
        return True, None

    if stage_number == 2:
        required_files = [config.USERS_JSON_PATH]

    elif stage_number == 3:
        required_files = [
            config.USERS_JSON_PATH,
            config.POSTS_JSON_PATH,
            config.FOLLOWS_JSON_PATH,
        ]

    # Validate existence
    missing = [str(p) for p in required_files if not os.path.exists(p)]

    if missing:
        message = f"Missing files: {', '.join(missing)}."
        logger.warning(f"Stage {stage_number} prerequisite check failed — {message}")
        return False, message

    logger.info(f"Stage {stage_number} prerequisite files found.")
    return True, None
