"""
Stage 5 UI helpers - Hybrid (Rich if available, plain-text fallback).

Functions:
 - list_all_users()
 - view_global_topics()
 - display_live_result(result_dict)    # new: pretty-prints live run payload or cached output
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, List

import pandas as pd

from src.core import utils
from src.core.logging_config import get_logger

logger = get_logger(__name__)

# Optional rich

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel        # ← FIXED (explicit import)
    from rich.text import Text
    from rich import box
    USE_RICH = True
    console = Console()
except Exception:
    USE_RICH = False
    console = None



# --- Helpers -----------------------------------------------------------
def _print_dataframe(df: pd.DataFrame, title: str):
    if USE_RICH:
        table = Table(title=title, box=box.MINIMAL_DOUBLE_HEAD)
        for col in df.columns:
            table.add_column(str(col), overflow="fold")
        for _, row in df.iterrows():
            table.add_row(*[str(x) if x is not None else "" for x in row.values])
        console.print(table)
    else:
        print(f"\n--- {title} ---")
        print(df.to_string(index=False))
        print("-" * 60)


# --- Primary view functions -------------------------------------------
def list_all_users():
    """List users stored in the database."""
    logger.info("Fetching all users...")
    engine = None
    try:
        engine = utils.get_sqlalchemy_engine()
        df = pd.read_sql("SELECT user_id, username, personas, subreddits FROM users ORDER BY user_id", engine)
        if df.empty:
            if USE_RICH:
                console.print("[yellow]No users found. Run Stage 1 and Stage 3 (ETL).[/yellow]")
            else:
                print("No users found. Run Stage 1 and Stage 3 (ETL).")
            return
        _print_dataframe(df, "All Users")
    except Exception as e:
        logger.exception("Error listing users: %s", e)
        if USE_RICH:
            console.print(f"[red]Error listing users: {e}[/red]")
        else:
            print(f"Error listing users: {e}")
    finally:
        if engine:
            engine.dispose()


def view_global_topics():
    """View global topics (batch analysis) with title, truncated summary, counts."""
    logger.info("Fetching global topic summaries...")
    engine = None
    try:
        engine = utils.get_sqlalchemy_engine()
        query = """
                SELECT gt.topic_id,
                       gt.topic_title,
                       gt.summary_text,
                       gt.sample_posts,
                       COUNT(ptm.post_id) AS post_count
                FROM global_topics gt
                         LEFT JOIN post_topic_mapping ptm ON gt.topic_id = ptm.topic_id
                GROUP BY gt.topic_id, gt.topic_title, gt.summary_text, gt.sample_posts
                ORDER BY post_count DESC; \
                """
        df = pd.read_sql(query, engine)
        if df.empty:
            if USE_RICH:
                console.print("[yellow]No topic summaries found. Run Stage 4 first.[/yellow]")
            else:
                print("No topic summaries found. Run Stage 4 first.")
            return
        
        # small cleanup for display
        df["summary_snippet"] = df["summary_text"].apply(
            lambda s: (s[:300] + "...") if s and len(s) > 300 else (s or ""))
        display_df = df[["topic_id", "post_count", "topic_title", "summary_snippet"]].rename(
            columns={"post_count": "Posts", "topic_title": "Title", "summary_snippet": "Summary"}
        )
        _print_dataframe(display_df, "Global Topic Summaries")
    except Exception as e:
        logger.exception("Error viewing global topics: %s", e)
        if USE_RICH:
            console.print(f"[red]Error viewing global topics: {e}[/red]")
        else:
            print(f"Error viewing global topics: {e}")
    finally:
        if engine:
            engine.dispose()


# --- Live result display ------------------------------------------------
def _format_bullets(bullets: Optional[List[str]]) -> str:
    if not bullets:
        return ""
    return "\n".join([f" • {b}" for b in bullets])


def display_live_result(result: dict):
    """
    Unified renderer for BOTH:
      - old global-mode output (flat)
      - new subreddit-wise hierarchical output
    """

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    if not result:
        console.print("[red]No result to display.[/red]")
        return

    status = result.get("status")
    payload = result.get("payload") or {}

    # ----------------------------
    # CASE 1 → NEW SUBREDDIT-WISE FLOW
    # ----------------------------
    if "subreddits" in payload:
        user = payload.get("user_id", "?")
        ts = payload.get("timestamp", "unknown")
        allocations = payload.get("allocations", {})

        console.rule(f"[bold cyan]{user} / reddit:subreddit_analysis[/bold cyan]")

        # Summary header
        console.print(
            Panel.fit(
                f"[bold]User:[/bold] {user}\n"
                f"[bold]Run at:[/bold] {ts}\n"
                f"[bold]Subreddits Analyzed:[/bold] {len(payload['subreddits'])}",
                title="Per-Subreddit Analysis", style="bold blue"
            )
        )

        # Table per subreddit
        for subreddit, info in payload["subreddits"].items():
            if "error" in info:
                console.print(f"[red]Error for r/{subreddit}: {info['error']}[/red]")
                continue

            allocated = info.get("allocated", 0)
            post_count = info.get("post_count", 0)
            topics = info.get("topics", [])

            sub_table = Table(title=f"r/{subreddit} — {post_count} posts (allocated {allocated})")

            sub_table.add_column("Topic #")
            sub_table.add_column("Title")
            sub_table.add_column("Bullets", overflow="fold")

            for t in topics:
                bullets = "\n".join(f"- {b}" for b in t.get("bullets", []))
                sub_table.add_row(str(t.get("topic_index")), t.get("title", ""), bullets)

            console.print(sub_table)

        return

    # ----------------------------
    # CASE 2 → OLD FLAT (GLOBAL) FLOW
    # ----------------------------
    post_count = payload.get("post_count", 0)
    ts = payload.get("created_at", "unknown")
    user = payload.get("user_id", "?")

    console.rule(f"[bold cyan]{user} / {payload.get('source','reddit')}[/bold cyan]")

    console.print(
        Panel.fit(
            f"[bold]User:[/bold] {user}\n"
            f"[bold]Posts processed:[/bold] {post_count}\n"
            f"[bold]Run at:[/bold] {ts}",
            title="Live Analysis Result"
        )
    )

    res = payload.get("result")
    if not res:
        console.print("[yellow]No summary returned.[/yellow]")
        return

    # Render topics (flat mode)
    table = Table(title="Topics")
    table.add_column("Index")
    table.add_column("Title")
    table.add_column("Bullets", overflow="fold")

    for t in res.get("topics", []):
        bullets = "\n".join(f"- {b}" for b in t.get("bullets", []))
        table.add_row(str(t.get("topic_index")), t.get("title", ""), bullets)

    console.print(table)


# Optional small test runner
if __name__ == "__main__":
    print("Testing Stage 5 UI helpers...")
    list_all_users()
    view_global_topics()
