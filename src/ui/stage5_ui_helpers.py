"""
Stage 5: UI Helpers
────────────────────────────────────────────
Provides user-facing database queries and
formatted console output for viewing:
  • All users
  • Global topic summaries (with titles)

Uses Rich for colorized output, with plain-text fallback.
"""

import logging
import pandas as pd

from src.core import utils

# Try to import rich (optional dependency)
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False
    console = None

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Helper: Print DataFrame as Rich Table or Plain Text
# ─────────────────────────────────────────────
def _print_dataframe(df: pd.DataFrame, title: str):
    """Pretty-print a DataFrame using rich if available, else plain text."""
    if USE_RICH:
        table = Table(title=title, box=box.MINIMAL_DOUBLE_HEAD)
        for col in df.columns:
            table.add_column(str(col), style="cyan", overflow="fold")
        for _, row in df.iterrows():
            table.add_row(*[str(x) if x is not None else "" for x in row.values])
        console.print(table)
    else:
        print(f"\n--- {title} ---")
        print(df.to_string(index=False))
        print("-" * 60)


# ─────────────────────────────────────────────
# Function: List all users
# ─────────────────────────────────────────────
def list_all_users():
    """Fetch and display all users from the database."""
    logger.info("Fetching all users from database...")
    engine = None
    try:
        engine = utils.get_sqlalchemy_engine()
        query = "SELECT user_id, username, personas, subreddits FROM users ORDER BY user_id;"
        df = pd.read_sql(query, engine)

        if df.empty:
            console.print("[yellow]⚠️ No users found. Run Stage 1 and ETL first.[/yellow]") if USE_RICH else \
                print("⚠️ No users found. Run Stage 1 and ETL first.")
            return

        _print_dataframe(df, "👤 All Users")

    except Exception as e:
        msg = f"Error listing users: {e}"
        logger.exception(msg)
        console.print(f"[red]🚨 {msg}[/red]") if USE_RICH else print(f"🚨 {msg}")

    finally:
        if engine:
            engine.dispose()
        logger.debug("Database connection closed.")


# ─────────────────────────────────────────────
# Function: View global topics
# ─────────────────────────────────────────────
def view_global_topics():
    """Fetch and display globally generated topics and summaries."""
    logger.info("Fetching global topic summaries...")
    engine = None
    try:
        engine = utils.get_sqlalchemy_engine()
        query = """
            SELECT gt.topic_id,
                   gt.topic_title,
                   gt.summary_text,
                   COUNT(ptm.post_id) AS post_count
            FROM global_topics gt
            LEFT JOIN post_topic_mapping ptm ON gt.topic_id = ptm.topic_id
            GROUP BY gt.topic_id, gt.topic_title, gt.summary_text
            ORDER BY post_count DESC;
        """
        df = pd.read_sql(query, engine)

        if df.empty:
            console.print("[yellow]⚠️ No topic summaries found. Run Stage 4 first.[/yellow]") if USE_RICH else \
                print("⚠️ No topic summaries found. Run Stage 4 first.")
            return

        if USE_RICH:
            table = Table(title="📈 Global Topic Summaries", box=box.SIMPLE_HEAVY)
            table.add_column("Topic ID", justify="center", style="bold cyan")
            table.add_column("Posts", justify="center", style="green")
            table.add_column("Title", style="magenta")
            table.add_column("Summary", style="white", overflow="fold")

            for _, row in df.iterrows():
                table.add_row(
                    str(row["topic_id"]),
                    str(row["post_count"]),
                    row.get("topic_title", "Untitled") or "Untitled",
                    row["summary_text"][:400] + ("..." if len(row["summary_text"]) > 400 else "")
                )

            console.print("\n")
            console.print(table)

        else:
            print("\n--- 📈 Global Topic Summaries ---")
            for _, row in df.iterrows():
                print("=" * 50)
                print(f"   TOPIC ID: {row['topic_id']} (Posts: {row['post_count']})")
                print(f"   TITLE    : {row.get('topic_title', 'Untitled')}")
                print(f"   SUMMARY  : {row['summary_text']}")
            print("=" * 50)

    except Exception as e:
        msg = f"Error viewing global topics: {e}"
        logger.exception(msg)
        console.print(f"[red]🚨 {msg}[/red]") if USE_RICH else print(f"🚨 {msg}")

    finally:
        if engine:
            engine.dispose()
        logger.debug("Database connection closed.")


# ─────────────────────────────────────────────
# Main (optional testing)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Stage 5 UI Helpers...\n")
    list_all_users()
    view_global_topics()
