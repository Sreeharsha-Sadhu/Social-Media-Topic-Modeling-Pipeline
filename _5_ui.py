# stage_5_ui_helpers.py

import pandas as pd
import utils

def list_all_users():
    """Fetches and displays all users from the database."""
    print("--- 👤 All Users ---")
    conn = None
    try:
        conn = utils.get_db_connection()
        df = pd.read_sql("SELECT user_id, username, personas FROM users ORDER BY user_id", conn)
        # Use print(df) for a standard console. Use display(df) if in IPython.
        print(df.to_string())
    except Exception as e:
        print(f"🚨 Error listing users: {e}")
        print("Have you run the 'Setup Database' and 'Run ETL' options yet?")
    finally:
        if conn:
            conn.close()

def view_results_for_user(user_id):
    """Fetches and displays the last analysis results for a user."""
    print(f"\n--- 📈 Viewing Analysis for: {user_id} ---")
    conn = None
    try:
        conn = utils.get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT topic_name, summary, post_count
                FROM user_topics
                WHERE user_id = %s
                ORDER BY post_count DESC
                """,
                (user_id,)
            )
            results = cur.fetchall()

            if not results:
                print(f"No results found for {user_id}.")
                print("Please run an analysis (Option 4) for this user first.")
                return

            for i, (topic_name, summary, post_count) in enumerate(results):
                print("\n" + "=" * 40)
                print(f"   TOPIC: {topic_name.split('_', 1)[1]}")
                print(f"   POSTS: {post_count}")
                print(f"   SUMMARY: {summary}")
            print("=" * 40)

    except Exception as e:
        print(f"🚨 Error viewing results: {e}")
    finally:
        if conn:
            conn.close()
