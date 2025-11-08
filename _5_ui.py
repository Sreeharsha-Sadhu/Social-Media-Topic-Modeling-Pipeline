# stage_5_ui_helpers.py
# (No changes to list_all_users function)

import pandas as pd
import utils


def list_all_users():
    """Fetches and displays all users from the database."""
    print("--- 👤 All Users ---")
    conn = None
    try:
        conn = utils.get_db_connection()
        df = pd.read_sql("SELECT user_id, username, personas FROM users ORDER BY user_id", conn)
        print(df.to_string())
    except Exception as e:
        print(f"🚨 Error listing users: {e}")
        print("Have you run the 'Setup Database' and 'Run ETL' options yet?")
    finally:
        if conn:
            conn.close()


# --- MODIFIED FUNCTION ---
def view_global_topics():
    """
    Fetches and displays the globally generated topics and their summaries.
    """
    print(f"\n--- 📈 Viewing Global Topic Summaries ---")
    conn = None
    try:
        conn = utils.get_db_connection()
        
        # New query to join global_topics with post_topic_mapping to get counts
        query = """
                SELECT gt.topic_id, \
                       gt.summary_text, \
                       COUNT(ptm.post_id) AS post_count
                FROM global_topics gt
                         LEFT JOIN post_topic_mapping ptm ON gt.topic_id = ptm.topic_id
                GROUP BY gt.topic_id, gt.summary_text
                ORDER BY post_count DESC; \
                """
        df = pd.read_sql(query, conn)
        
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
        if conn:
            conn.close()
