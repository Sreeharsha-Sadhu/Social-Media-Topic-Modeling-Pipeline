"""
config.py
----------------------------------
Central configuration module for the MMDS Project.
Handles environment variables, path setup, and runtime configuration
for both Spark and local Pandas-based workflows.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------------------

# Locate and load the .env file
load_dotenv()

# -------------------------------------------------------------------
# Base Directories
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# Database Configuration
# -------------------------------------------------------------------

DB_NAME = os.getenv("DB_NAME", "mmds")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "12345678")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# -------------------------------------------------------------------
# Reddit API Configuration
# -------------------------------------------------------------------

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "mmds-reddit-client")

# -------------------------------------------------------------------
# Data File Paths
# -------------------------------------------------------------------

USERS_JSON_PATH = DATA_DIR / "users.json"
FOLLOWS_JSON_PATH = DATA_DIR / "follows.json"
POSTS_JSON_PATH = DATA_DIR / "posts.json"
EDGELIST_PATH = DATA_DIR / "follows.edgelist"

# -------------------------------------------------------------------
# Spark Configuration
# -------------------------------------------------------------------

USE_SPARK_ETL = os.getenv("USE_SPARK_ETL", "false").lower() == "true"
USE_SPARK_ANALYSIS = os.getenv("USE_SPARK_ANALYSIS", "false").lower() == "true"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_APP_NAME = os.getenv("SPARK_APP_NAME", "SocialMediaETL")
SPARK_ANALYSIS_TOPICS = int(os.getenv("SPARK_ANALYSIS_TOPICS", "20"))

# -------------------------------------------------------------------
# Logging Configuration Hook
# -------------------------------------------------------------------

def get_log_file_path(filename: str = "mmds.log") -> Path:
    """
    Returns the full path for the primary project log file.
    Creates a 'logs' directory under project root if not existing.
    """
    return LOG_DIR / filename
