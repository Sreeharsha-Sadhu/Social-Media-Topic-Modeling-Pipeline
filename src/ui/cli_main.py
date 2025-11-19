"""
cli_main.py — Reddit-only CLI for Live Topic Modelling
───────────────────────────────────────────────────────────
This CLI reflects the new architecture:

✔ No synthetic posts
✔ PRAW-first Reddit ingestion (with simulator fallback)
✔ Adaptive per-subreddit topic modelling
✔ Global live analysis (legacy)
✔ ETL only creates schema + optional user seeding
"""

from __future__ import annotations

import sys
import logging
import click

from src.core import utils
from src.core.logging_config import get_logger

# Pipeline stages
from src.data_pipeline import (
    stage3_etl,
    stage4_global_analysis,
    stage1_generate_users,
    stage2_generate_posts,  # Only for synthetic user seeding (no posts)
)
from src.ui import stage5_ui_helpers

# Live Reddit fetcher
try:
    from src.live.live_reddit import (
        fetch_reddit_user_posts,                # flat/global mode
        analyze_reddit_feed_per_subreddit       # adaptive per-subreddit mode
    )
except Exception:
    fetch_reddit_user_posts = None
    analyze_reddit_feed_per_subreddit = None

# Rich UI (optional)
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel

    console = Console()
    USE_RICH = True
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
except Exception:
    USE_RICH = False
    console = None
    logging.basicConfig(level=logging.INFO)

logger = get_logger(__name__)


# ------------------------
# Utility helpers
# ------------------------
def confirm(prompt: str) -> bool:
    return input(f"{prompt} (y/n): ").lower().strip() == "y"


def pause():
    input("\nPress Enter to return...")


# ------------------------
# ETL / Setup
# ------------------------
def cli_setup_database():
    utils.clear_screen()
    if not confirm("This will DROP schema tables. Continue?"):
        return
    stage3_etl.create_tables()
    stage4_global_analysis.setup_database_tables()
    print("Database successfully initialized.")
    pause()


def cli_seed_users():
    utils.clear_screen()
    if confirm("Run Stage 1: Generate Users?"):
        stage1_generate_users.main()
    if confirm("Run Stage 2: Generate Follows only? (Posts ignored)"):
        stage2_generate_posts.main()
    pause()


def cli_run_etl():
    utils.clear_screen()
    print("ETL only creates schema and (optionally) seeds users.")
    seed = confirm("Seed users from JSON files if available?")
    stage3_etl.run_etl(seed_users=seed)
    pause()


# ------------------------
# Live Reddit Analysis
# ------------------------
def cli_run_reddit_global():
    utils.clear_screen()
    print("Live Reddit Global Summary")

    if fetch_reddit_user_posts is None:
        print("Reddit fetcher unavailable.")
        return pause()

    user_id = input("Enter user ID (e.g., user_1): ").strip()
    top_n_raw = input("Fetch top N posts (default 50): ").strip()
    top_n = int(top_n_raw) if top_n_raw.isdigit() else 50

    if USE_RICH:
        with console.status("[green]Running global Reddit analysis...[/green]"):
            res = stage4_global_analysis.run_live_analysis_for_user(
                user_id=user_id,
                source="reddit-global",
                fetcher_callable=fetch_reddit_user_posts,
                top_n=top_n,
            )
    else:
        print("Running global Reddit analysis...")
        res = stage4_global_analysis.run_live_analysis_for_user(
            user_id=user_id,
            source="reddit-global",
            fetcher_callable=fetch_reddit_user_posts,
            top_n=top_n,
        )

    utils.clear_screen()
    stage5_ui_helpers.display_live_result(res)
    pause()


def cli_run_reddit_subreddit():
    utils.clear_screen()
    print("Adaptive Per-Subreddit Reddit Analysis (Recommended)")

    if analyze_reddit_feed_per_subreddit is None:
        print("Subreddit analyzer unavailable.")
        return pause()

    user_id = input("Enter user ID (e.g., user_1): ").strip()
    total_raw = input("Total adaptive posts to fetch (default 100): ").strip()
    total = int(total_raw) if total_raw.isdigit() else 100

    if USE_RICH:
        with console.status("[cyan]Running per-subreddit Reddit analysis...[/cyan]"):
            res = stage4_global_analysis.run_subreddit_topic_analysis(
                user_id=user_id,
                top_n_total=total,
                prefer_since_last_run=True,
            )
    else:
        print("Running per-subreddit Reddit analysis...")
        res = stage4_global_analysis.run_subreddit_topic_analysis(
            user_id=user_id,
            top_n_total=total,
            prefer_since_last_run=True,
        )

    utils.clear_screen()
    stage5_ui_helpers.display_live_result(res)
    pause()


# ------------------------
# Admin / UI navigation
# ------------------------
def cli_view_global_topics():
    utils.clear_screen()
    stage5_ui_helpers.view_global_topics()
    pause()


def cli_list_users():
    utils.clear_screen()
    stage5_ui_helpers.list_all_users()
    pause()


# ------------------------
# Interactive Menu
# ------------------------
def main_menu():
    while True:
        utils.clear_screen()

        if USE_RICH:
            console.rule("[bold magenta]Reddit Live Topic Modeller[/bold magenta]")
            console.print("[cyan]1[/cyan] → Setup Database (Schema Only)")
            console.print("[cyan]2[/cyan] → Seed Users (Optional)")
            console.print("[cyan]3[/cyan] → ETL (Schema + Optional Users)")
            console.print("[cyan]4[/cyan] → View Global Topics (Batch Results)")
            console.print("[cyan]5[/cyan] → List Users")
            console.print("[cyan]6[/cyan] → Live Reddit Global Summary (Flat Mode)")
            console.print("[cyan]7[/cyan] → Live Reddit Per-Subreddit Analysis (Adaptive Recommended)")
            console.print("[cyan]x[/cyan] → Exit")
        else:
            print("──── Reddit Live Topic Modeller ────")
            print("1 → Setup Database")
            print("2 → Seed Users")
            print("3 → ETL (Schema Updates Only)")
            print("4 → View Global Topics")
            print("5 → List Users")
            print("6 → Live Global Reddit Summary")
            print("7 → Live Per-Subreddit Reddit Summary")
            print("x → Exit")

        choice = input("\nEnter choice: ").strip().lower()

        if choice == "1":
            cli_setup_database()
        elif choice == "2":
            cli_seed_users()
        elif choice == "3":
            cli_run_etl()
        elif choice == "4":
            cli_view_global_topics()
        elif choice == "5":
            cli_list_users()
        elif choice == "6":
            cli_run_reddit_global()
        elif choice == "7":
            cli_run_reddit_subreddit()
        elif choice == "x":
            print("Goodbye!")
            sys.exit(0)
        else:
            input("Invalid choice — press Enter to retry...")


# ------------------------
# Click commands
# ------------------------
@click.group()
def cli():
    """Reddit Live Topic Modeller"""
    pass


@cli.command()
def menu():
    """Launch interactive menu"""
    main_menu()


@cli.command()
def setup():
    """Setup DB schema only"""
    cli_setup_database()


@cli.command()
def live_global():
    """Run global-mode Reddit summary"""
    cli_run_reddit_global()


@cli.command()
def live_subreddit():
    """Run per-subreddit adaptive Reddit analysis"""
    cli_run_reddit_subreddit()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli()
    else:
        main_menu()
