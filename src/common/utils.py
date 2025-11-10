# utils.py

import os

import psycopg2
from IPython.display import clear_output
from dotenv import load_dotenv
from sqlalchemy import create_engine

_spark_session = None
load_dotenv()

# utils.py  (add/replace get_spark_session implementation)

import sys
from pyspark.sql import SparkSession
from pyspark import SparkConf
from src.config import settings


def get_spark_session():
    """
    Returns a singleton SparkSession configured for our project.
    Adds a few safety configs (faulthandler, worker reuse, memory),
    and ensures Python executable is consistent across driver/executors.
    """
    global _spark_session
    if _spark_session is None:
        print(f"🧩 Initializing SparkSession for {settings.SPARK_APP_NAME} on {settings.SPARK_MASTER}...")
        
        conf = SparkConf()
        
        # Basic app/master
        conf.setAppName(settings.SPARK_APP_NAME)
        conf.setMaster(settings.SPARK_MASTER)
        
        # Helpful configs for PySpark stability
        conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        conf.set("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
        conf.set("spark.python.worker.reuse", "true")
        conf.set("spark.default.parallelism", "4")  # don’t oversubscribe cores
        conf.set("spark.sql.shuffle.partitions", "4")  # fewer shuffle tasks
        conf.set("spark.executor.memory", "3g")
        conf.set("spark.driver.memory", "6g")
        conf.set("spark.python.worker.memory", "512m")
        conf.set("spark.python.profile", "true")  # enables extra diagnostics
        
        # Make sure executors use the same python interpreter as the driver.
        # This reduces Python-version/path mismatch crashes.
        python_exe = sys.executable
        try:
            conf.setExecutorEnv("PYSPARK_PYTHON", python_exe)
            conf.setExecutorEnv("PYSPARK_DRIVER_PYTHON", python_exe)
        except Exception:
            # older pyspark versions may not have setExecutorEnv; setting options below instead
            conf.set("spark.executorEnv.PYSPARK_PYTHON", python_exe)
            conf.set("spark.executorEnv.PYSPARK_DRIVER_PYTHON", python_exe)
        
        # optional: reduce logging noise
        # conf.set("spark.ui.showConsoleProgress", "false")
        
        _spark_session = SparkSession.builder.config(conf=conf).getOrCreate()
        # set log level to WARN to reduce noise
        _spark_session.sparkContext.setLogLevel("WARN")
        print("✅ SparkSession started")
    return _spark_session


def get_db_connection():
    """Establishes a new (psycopg2) connection to the PostgreSQL database."""
    conn_string = f"dbname='{settings.DB_NAME}' user='{settings.DB_USER}' password='{settings.DB_PASS}' host='{settings.DB_HOST}' port='{settings.DB_PORT}'"
    return psycopg2.connect(conn_string)


def get_sqlalchemy_engine():
    """Establishes a new (sqlalchemy) engine for pandas.to_sql()."""
    conn_string = f"postgresql+psycopg2://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
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
        if not os.path.exists(settings.USERS_JSON_PATH):
            return False, f"Missing file: {settings.USERS_JSON_PATH}. Please run Stage 1."
        return True, None
    
    if stage_number == 3:
        missing = []
        if not os.path.exists(settings.USERS_JSON_PATH):
            missing.append(settings.USERS_JSON_PATH)
        if not os.path.exists(settings.POSTS_JSON_PATH):
            missing.append(settings.POSTS_JSON_PATH)
        if not os.path.exists(settings.FOLLOWS_JSON_PATH):
            missing.append(settings.FOLLOWS_JSON_PATH)
        
        if missing:
            return False, f"Missing files: {', '.join(missing)}. Please run Stages 1 and 2."
        return True, None
    
    return True, None
