# config.py

import os
from dotenv import load_dotenv

# --- Load Environment Variables ---
# This line finds the .env file in the project root
# and loads its variables into os.environ
load_dotenv()

# --- Database Configuration ---
# We now read the values from the environment
DB_NAME = os.getenv("DB_NAME", "mmds")
DB_USER = os.getenv("DB_USER", "sadhu")
DB_PASS = os.getenv("DB_PASS", "12345678")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# --- Reddit API Configuration ---
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# --- Project Structure (Unchanged) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- File Paths (Unchanged) ---
USERS_JSON_PATH = os.path.join(DATA_DIR, "users.json")
EDGELIST_PATH = os.path.join(DATA_DIR, "follows.edgelist")
FOLLOWS_JSON_PATH = os.path.join(DATA_DIR, "follows.json")
POSTS_JSON_PATH = os.path.join(DATA_DIR, "posts.json")
