# config.py

import os

# --- Database Configuration ---
DB_NAME = "mmds"
DB_USER = "sadhu"
DB_PASS = "12345678"
DB_HOST = "localhost"
DB_PORT = "5432"

# --- Project Structure ---
# Get the absolute path of the directory this file is in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory
DATA_DIR = os.path.join(BASE_DIR, "data")

# --- File Paths ---
USERS_JSON_PATH = os.path.join(DATA_DIR, "users.json")
EDGELIST_PATH = os.path.join(DATA_DIR, "follows.edgelist")
FOLLOWS_JSON_PATH = os.path.join(DATA_DIR, "follows.json")
POSTS_JSON_PATH = os.path.join(DATA_DIR, "posts.json")
