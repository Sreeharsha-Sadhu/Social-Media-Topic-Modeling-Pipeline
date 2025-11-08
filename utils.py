# utils.py

import os
import psycopg2
from sqlalchemy import create_engine
from IPython.display import clear_output
import config
from pyspark.sql import SparkSession
from pyspark import SparkConf

_spark_session = None

def get_spark_session():
    """Initializes (or reuses) a SparkSession based on config."""
    global _spark_session
    if _spark_session is None:
        print(f"🧩 Initializing SparkSession for {config.SPARK_APP_NAME}...")
        conf = (
            SparkConf()
            .setAppName(config.SPARK_APP_NAME)
            .setMaster(config.SPARK_MASTER)
            .set("spark.sql.execution.arrow.pyspark.enabled", "true")
            .set("spark.sql.repl.eagerEval.enabled", "true")
            .set("spark.driver.memory", "4g")
        )
        _spark_session = SparkSession.builder.config(conf=conf).getOrCreate()
        print(f"✅ SparkSession started on {config.SPARK_MASTER}")
    return _spark_session

def get_db_connection():
    """Establishes a new (psycopg2) connection to the PostgreSQL database."""
    conn_string = f"dbname='{config.DB_NAME}' user='{config.DB_USER}' password='{config.DB_PASS}' host='{config.DB_HOST}' port='{config.DB_PORT}'"
    return psycopg2.connect(conn_string)


def get_sqlalchemy_engine():
    """Establishes a new (sqlalchemy) engine for pandas.to_sql()."""
    conn_string = f"postgresql+psycopg2://{config.DB_USER}:{config.DB_PASS}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
    return create_engine(conn_string)


def clear_screen():
    """Clears the console output."""
    try:
        clear_output(wait=True)
    except ImportError:
        os.system('cls' if os.name == 'nt' else 'clear')


def check_file_prerequisites(stage_number):
    """
    Checks if the required files for a given stage exist.
    Returns (True, None) if successful, or (False, error_message) if not.
    """
    if stage_number == 1:
        return True, None
    
    if stage_number == 2:
        if not os.path.exists(config.USERS_JSON_PATH):
            return False, f"Missing file: {config.USERS_JSON_PATH}. Please run Stage 1."
        return True, None
    
    if stage_number == 3:
        missing = []
        if not os.path.exists(config.USERS_JSON_PATH):
            missing.append(config.USERS_JSON_PATH)
        if not os.path.exists(config.POSTS_JSON_PATH):
            missing.append(config.POSTS_JSON_PATH)
        if not os.path.exists(config.FOLLOWS_JSON_PATH):
            missing.append(config.FOLLOWS_JSON_PATH)
        
        if missing:
            return False, f"Missing files: {', '.join(missing)}. Please run Stages 1 and 2."
        return True, None
    
    return True, None
