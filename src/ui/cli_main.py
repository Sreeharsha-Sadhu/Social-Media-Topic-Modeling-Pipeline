"""
CLI + Menu entrypoint for the Social Media Topic & Summary Tool.

Provides:
 - Interactive menu (keyboard-driven)
 - Click-based CLI for scripted usage

Uses Rich for nicer output when available; falls back to plain text.
"""

from __future__ import annotations

import sys
import logging
from typing import Optional

import click

from src.core import utils
from src.core.logging_config import get_logger

# pipeline modules (existing project)
from src.data_pipeline import stage1_generate_users, stage2_generate_posts, stage3_etl, stage4_global_analysis
from src.ui import stage5_ui_helpers

logger = get_logger(__name__)

# Try to import optional live analyzers (these were uploaded by you)
try:
    from src.live.live_reddit import fetch_reddit_user_posts
except Exception:
    fetch_reddit_user_posts = None
try:
    from src.live.live_linkedin import fetch_linkedin_user_posts
except Exception:
    fetch_linkedin_user_posts = None

# Try to import Rich
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    console = Console()
    USE_RICH = True
except Exception:
    USE_RICH = False
    console = None

# configure logging (uses your logging_config.get_logger already)
if USE_RICH:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, handlers=[RichHandler()])
else:
    logging.basicConfig(level=logging.INFO)

# utility
def confirm_continue(prompt: str) -> bool:
    return input(f"{prompt} (y/n): ").strip().lower() == "y"

# --- Stage wrappers -----------------------------------------------------
def run_stage_1():
    utils.clear_screen()
    stage1_generate_users.main()
    input("\nPress Enter to return...")

def run_stage_2():
    utils.clear_screen()
    ok, msg = utils.check_file_prerequisites(2)
    if not ok:
        if USE_RICH:
            console.print(f"[red]Prerequisite failed:[/red] {msg}")
        else:
            print(f"Prerequisite failed: {msg}")
        if confirm_continue("Run Stage 1 now?"):
            run_stage_1()
            utils.clear_screen()
            stage2_generate_posts.main()
    else:
        stage2_generate_posts.main()
    input("\nPress Enter to return...")

def run_stage_3(mode: str = "pandas"):
    utils.clear_screen()
    if mode == "spark":
        stage3_etl.run_etl()
    else:
        # Pandas ETL
        stage3_etl._run_etl_pandas()
    input("\nPress Enter to return...")

def run_stage_4_interactive():
    utils.clear_screen()
    if USE_RICH:
        console.print(Panel("[bold cyan]Launch Global Topic Model (Batch AI)[/bold cyan]"))
    else:
        print("Launch Global Topic Model (Batch AI)")
    if not confirm_continue("Continue?"):
        return
    stage4_global_analysis.run_global_analysis()
    input("\nPress Enter to return...")

def run_stage_5():
    utils.clear_screen()
    stage5_ui_helpers.view_global_topics()
    input("\nPress Enter to return...")

def list_users():
    utils.clear_screen()
    stage5_ui_helpers.list_all_users()
    input("\nPress Enter to return...")

# --- Live analysis helpers ---------------------------------------------
def _perform_live_reddit_analysis_prompt():
    utils.clear_screen()
    print("Live Reddit Topic Analysis")
    user_id = input("Enter the User ID (e.g., user_1): ").strip()
    if not user_id:
        print("User ID required.")
        input("\nPress Enter to return...")
        return
    top_n_raw = input("Fetch top N posts (default 50): ").strip()
    top_n = int(top_n_raw) if top_n_raw.isdigit() else 50

    if fetch_reddit_user_posts is None:
        msg = "Reddit fetcher not available. Ensure src.live.live_reddit is present."
        if USE_RICH:
            console.print(f"[red]{msg}[/red]")
        else:
            print(msg)
        input("\nPress Enter to return...")
        return

    if USE_RICH:
        with console.status("[bold green]Running live Reddit analysis...[/bold green]"):
            res = stage4_global_analysis.run_live_analysis_for_user(
                user_id=user_id,
                source="reddit",
                fetcher_callable=fetch_reddit_user_posts,
                top_n=top_n
            )
    else:
        print("Running live Reddit analysis...")
        res = stage4_global_analysis.run_live_analysis_for_user(
            user_id=user_id,
            source="reddit",
            fetcher_callable=fetch_reddit_user_posts,
            top_n=top_n
        )

    utils.clear_screen()
    stage5_ui_helpers.display_live_result(res)
    input("\nPress Enter to return...")

def _perform_live_linkedin_analysis_prompt():
    utils.clear_screen()
    print("Live LinkedIn Topic Analysis")
    user_id = input("Enter the User ID (e.g., user_1): ").strip()
    if not user_id:
        print("User ID required.")
        input("\nPress Enter to return...")
        return
    top_n_raw = input("Fetch top N posts (default 50): ").strip()
    top_n = int(top_n_raw) if top_n_raw.isdigit() else 50

    if fetch_linkedin_user_posts is None:
        msg = "LinkedIn fetcher not available. Ensure src.live.live_linkedin is present."
        if USE_RICH:
            console.print(f"[red]{msg}[/red]")
        else:
            print(msg)
        input("\nPress Enter to return...")
        return

    if USE_RICH:
        with console.status("[bold green]Running live LinkedIn analysis...[/bold green]"):
            res = stage4_global_analysis.run_live_analysis_for_user(
                user_id=user_id,
                source="linkedin",
                fetcher_callable=fetch_linkedin_user_posts,
                top_n=top_n
            )
    else:
        print("Running live LinkedIn analysis...")
        res = stage4_global_analysis.run_live_analysis_for_user(
            user_id=user_id,
            source="linkedin",
            fetcher_callable=fetch_linkedin_user_posts,
            top_n=top_n
        )

    utils.clear_screen()
    stage5_ui_helpers.display_live_result(res)
    input("\nPress Enter to return...")

# --- Interactive main menu ---------------------------------------------
def main_menu():
    while True:
        utils.clear_screen()
        if USE_RICH:
            console.rule("[bold magenta] Social Media Topic & Summary Tool [/bold magenta]")
            console.print("[cyan]1[/cyan] → Setup Database")
            console.print("[cyan]2[/cyan] → Run Stage 1: Generate Users")
            console.print("[cyan]3[/cyan] → Run Stage 2: Generate Posts")
            console.print("[cyan]4[/cyan] → Run Stage 3: ETL")
            console.print("[cyan]5[/cyan] → Run Stage 4: Global Topic Model")
            console.print("[cyan]6[/cyan] → Run Stage 5: View Global Topics")
            console.print("[cyan]7[/cyan] → List All Users")
            console.print("[cyan]8[/cyan] → Live Reddit Analysis")
            console.print("[cyan]9[/cyan] → Live LinkedIn Analysis")
            console.print("[cyan]x[/cyan] → Exit")
        else:
            print("────────────────────── Social Media Topic & Summary Tool ──────────────────────")
            print("1 → Setup Database")
            print("2 → Run Stage 1: Generate Users")
            print("3 → Run Stage 2: Generate Posts")
            print("4 → Run Stage 3: ETL")
            print("5 → Run Stage 4: Global Topic Model")
            print("6 → Run Stage 5: View Global Topics")
            print("7 → List All Users")
            print("8 → Live Reddit Analysis")
            print("9 → Live LinkedIn Analysis")
            print("x → Exit")

        choice = input("\nEnter your choice: ").strip().lower()

        if choice == "1":
            utils.clear_screen()
            if confirm_continue("This will DROP some analysis tables. Continue?"):
                stage3_etl.create_tables()
                stage4_global_analysis.setup_database_tables()
                if USE_RICH:
                    console.print("[green]Database setup complete.[/green]")
                else:
                    print("Database setup complete.")
                input("\nPress Enter to return...")
        elif choice == "2":
            run_stage_1()
        elif choice == "3":
            run_stage_2()
        elif choice == "4":
            # choose ETL mode if spark available
            if confirm_continue("Run Spark ETL? (Otherwise Pandas)"):
                run_stage_3(mode="spark")
            else:
                run_stage_3(mode="pandas")
        elif choice == "5":
            run_stage_4_interactive()
        elif choice == "6":
            run_stage_5()
        elif choice == "7":
            list_users()
        elif choice == "8":
            _perform_live_reddit_analysis_prompt()
        elif choice == "9":
            _perform_live_linkedin_analysis_prompt()
        elif choice == "x":
            if USE_RICH:
                console.print("[bold red]Exiting... Goodbye![/bold red]")
            else:
                print("Exiting... Goodbye!")
            sys.exit(0)
        else:
            input("Invalid choice. Press Enter to try again...")

# --- Click CLI ---------------------------------------------------------
@click.group()
def cli():
    """Social Media Topic & Summary Tool (CLI)."""
    pass

@cli.command()
def menu():
    """Launch interactive menu."""
    main_menu()

@cli.command()
@click.option("--etl", type=click.Choice(["pandas","spark"]), default="pandas")
def etl(etl):
    """Run ETL (Stage 3)."""
    run_stage_3(mode=etl)

@cli.command()
def batch_analysis():
    """Run batch global analysis (Stage 4)."""
    stage4_global_analysis.run_global_analysis()

@cli.command()
@click.option("--user", required=True, help="User id (e.g., user_1)")
@click.option("--source", type=click.Choice(["reddit","linkedin"]), default="reddit")
@click.option("--top_n", default=50, help="Max posts to fetch (default 50)")
def live(user: str, source: str, top_n: int):
    """Run live analysis for a specific user+source."""
    if source == "reddit":
        fetcher = fetch_reddit_user_posts
    else:
        fetcher = fetch_linkedin_user_posts

    if fetcher is None:
        print(f"Fetcher for {source} not available.")
        return

    if USE_RICH:
        with console.status(f"[bold green]Running live {source} analysis for {user}...[/bold green]"):
            res = stage4_global_analysis.run_live_analysis_for_user(
                user_id=user, source=source, fetcher_callable=fetcher, top_n=top_n
            )
    else:
        print(f"Running live {source} analysis for {user}...")
        res = stage4_global_analysis.run_live_analysis_for_user(
            user_id=user, source=source, fetcher_callable=fetcher, top_n=top_n
        )

    stage5_ui_helpers.display_live_result(res)

# Entrypoint
if __name__ == "__main__":
    # If command-line args provided => click CLI, else launch menu
    if len(sys.argv) > 1:
        cli()
    else:
        main_menu()
