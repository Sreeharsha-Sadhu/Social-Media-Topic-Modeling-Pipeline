# utils.py

import os
import psycopg2
from IPython.display import clear_output
import config


def get_db_connection():
    """Establishes a new connection to the PostgreSQL database."""
    conn_string = f"dbname='{config.DB_NAME}' user='{config.DB_USER}' password='{config.DB_PASS}' host='{config.DB_HOST}' port='{config.DB_PORT}'"
    return psycopg2.connect(conn_string)


def clear_screen():
    """Clears the console output."""
    try:
        # This works in Jupyter, VSCode notebooks, and IPython consoles
        clear_output(wait=True)
    except ImportError:
        # Fallback for standard Python terminals
        os.system('cls' if os.name == 'nt' else 'clear')


def check_file_prerequisites(stage_number):
    """
    Checks if the required files for a given stage exist.
    Returns (True, None) if successful, or (False, error_message) if not.
    """
    if stage_number == 1:
        # Stage 1 has no file prerequisites
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
    
    # Stages 4 & 5 depend on the database, not files.
    # We'll check for DB tables in their own setup.
    return True, None
