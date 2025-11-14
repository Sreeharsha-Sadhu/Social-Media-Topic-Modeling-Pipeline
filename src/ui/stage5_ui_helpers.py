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


def display_live_result(result: Dict[str, Any]):
    """
    Pretty-print a live-run payload produced by stage4.run_live_analysis_for_user (or cached payload).
    Expected formats:
      {"status":"ok","payload": { "user_id":..., "source":..., "created_at":..., "post_count":..., "result": {...}, "post_ids":[...] } }
      or cached fallback variations.
    """
    if not result:
        print("No result to display.")
        return
    
    status = result.get("status", "unknown")
    payload = result.get("payload") or result.get("result") or result.get("data") or None
    
    if payload is None:
        # maybe result itself is the payload
        if "user_id" in result:
            payload = result
        else:
            if USE_RICH:
                console.print("[yellow]No payload available in result.[/yellow]")
            else:
                print("No payload available in result.")
            return
    
    user_id = payload.get("user_id", "unknown")
    source = payload.get("source", "unknown")
    created_at = payload.get("created_at", "unknown")
    post_count = payload.get("post_count", 0)
    res = payload.get("result") or payload.get("topics") or payload.get("analysis") or {}
    
    # top-level header
    header_text = f"Live Analysis Result\nUser: {user_id}\nSource: {source}\nPosts processed: {post_count}\nRun at: {created_at}"
    if USE_RICH:
        console.rule("[bold green]Live Analysis Result[/bold green]")
        console.print(Panel(header_text, title=f"[cyan]{user_id} / {source}[/cyan]"))
    else:
        print("Live Analysis Result")
        print("-" * 60)
        print(header_text)
        print("-" * 60)
    
    # result payload rendering
    rtype = res.get("type") if isinstance(res, dict) else None
    
    if rtype == "single_summary" or (isinstance(res, dict) and res.get("topics") and len(res.get("topics", [])) == 1):
        # single-topic summary
        topic = res["topics"][0] if "topics" in res else res
        title = topic.get("title") or topic.get("topic_title") or "Untitled"
        summary = topic.get("summary") or topic.get("summary_text") or ""
        sample_posts = topic.get("sample_posts") or []
        # optional additional fields (sentiment, bullets) if present in payload
        sentiment = topic.get("sentiment")
        bullets = topic.get("key_points") or topic.get("bullets")
        
        if USE_RICH:
            summary_text = Text(summary)
            panel = Panel.fit(f"[bold]{title}[/bold]\n\n{summary_text}\n\n{_format_bullets(bullets)}",
                              title="Topic 0", border_style="magenta")
            console.print(panel)
            if sample_posts:
                console.print(
                    Panel("\n".join([f"- {s}" for s in sample_posts]), title="Sample Posts", border_style="blue"))
            if sentiment:
                console.print(f"[bold]Sentiment:[/bold] {sentiment}")
        else:
            print(f"\nTopic 0")
            print(f"Title: {title}")
            print(f"Summary:\n{summary}\n")
            if bullets:
                print("Key points:")
                print(_format_bullets(bullets))
            if sample_posts:
                print("\nSample Posts:")
                for s in sample_posts:
                    print(f" - {s}")
    
    elif rtype == "clustered" or (isinstance(res, dict) and res.get("topics")):
        topics = res.get("topics", [])
        for t in topics:
            tidx = t.get("topic_index") or t.get("topic_id") or t.get("topic_index") or 0
            title = t.get("title") or "Untitled"
            summary = t.get("summary") or ""
            sample_posts = t.get("sample_posts") or []
            bullets = t.get("key_points") or t.get("key_bullets") or t.get("sample_bullets") or None
            sentiment = t.get("sentiment")
            
            if USE_RICH:
                panel = Panel.fit(f"[bold]{title}[/bold]\n\n{summary}\n\n{_format_bullets(bullets)}",
                                  title=f"Topic {tidx}", border_style="magenta")
                console.print(panel)
                if sample_posts:
                    console.print(
                        Panel("\n".join([f"- {s}" for s in sample_posts]), title="Sample Posts", border_style="blue"))
                if sentiment:
                    console.print(f"[bold]Sentiment:[/bold] {sentiment}")
            else:
                print(f"\nTopic {tidx}")
                print(f"Title: {title}")
                print(f"Summary:\n{summary}\n")
                if bullets:
                    print("Key points:")
                    print(_format_bullets(bullets))
                if sample_posts:
                    print("Sample posts:")
                    for s in sample_posts:
                        print(f" - {s}")
    
    else:
        # unknown/residual format — pretty-print available keys
        if USE_RICH:
            console.print_json(json.dumps(res))
        else:
            print("Result (raw):")
            print(json.dumps(res, indent=2))
    
    # footer: show cached id if present
    output_id = payload.get("output_id")
    if output_id:
        if USE_RICH:
            console.print(f"[dim]Stored output_id: {output_id}[/dim]")
        else:
            print(f"[Stored output_id: {output_id}]")


# Optional small test runner
if __name__ == "__main__":
    print("Testing Stage 5 UI helpers...")
    list_all_users()
    view_global_topics()
