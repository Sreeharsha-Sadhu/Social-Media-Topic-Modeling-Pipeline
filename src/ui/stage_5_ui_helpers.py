# stage_5_ui_helpers.py

import pandas as pd

from src.common import utils


def list_all_users():
    """Fetches and displays all users from the database."""
    print("--- 👤 All Users ---")
    engine = None
    try:
        # --- FIX: Use SQLAlchemy engine for Pandas ---
        engine = utils.get_sqlalchemy_engine()
        df = pd.read_sql("SELECT user_id, username, personas, subreddits FROM users ORDER BY user_id", engine)
        print(df.to_string())
    except Exception as e:
        print(f"🚨 Error listing users: {e}")
        print("Have you run the 'Setup Database' and 'Run ETL' options yet?")
    finally:
        if engine:
            engine.dispose()


def view_global_topics():
    """
    Fetches and displays the globally generated topics and their summaries.
    """
    print(f"\n--- 📈 Viewing Global Topic Summaries ---")
    engine = None
    try:
        # --- FIX: Use SQLAlchemy engine for Pandas ---
        engine = utils.get_sqlalchemy_engine()
        
        query = """
                SELECT gt.topic_id, \
                       gt.summary_text, \
                       COUNT(ptm.post_id) AS post_count
                FROM global_topics gt
                         LEFT JOIN post_topic_mapping ptm ON gt.topic_id = ptm.topic_id
                GROUP BY gt.topic_id, gt.summary_text
                ORDER BY post_count DESC; \
                """
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print("No results found. Please run 'Stage 4: Generate Global Topic Model' first.")
            return
        
        for index, row in df.iterrows():
            print("\n" + "=" * 40)
            print(f"   TOPIC ID: {row['topic_id']} (Posts: {row['post_count']})")
            print(f"   SUMMARY: {row['summary_text']}")
        print("=" * 40)
    
    except Exception as e:
        print(f"🚨 Error viewing results: {e}")
    finally:
        if engine:
            engine.dispose()
