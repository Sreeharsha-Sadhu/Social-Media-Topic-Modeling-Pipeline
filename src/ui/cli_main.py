"""
CLI and Menu Entry Point
──────────────────────────────────────────────
Now includes:
  • Menu-based UI
  • Click-based CLI
  • Integrated Reddit & LinkedIn Live Analysis
"""

import sys
import logging
import click
from rich.console import Console
from rich.logging import RichHandler

from src.core import utils
from src.data_pipeline import (
    stage1_generate_users,
    stage2_generate_posts,
    stage3_etl,
    stage4_global_analysis,
)
from src.ui import stage5_ui_helpers

# Live fetchers (new)
from src.live.live_reddit import fetch_reddit_user_posts
from src.live.live_linkedin import fetch_linkedin_user_posts

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True)],
)
logger = logging.getLogger("CLI")


# ─────────────────────────────────────────────
# Helper: Confirmation
# ─────────────────────────────────────────────
def confirm_continue(prompt: str) -> bool:
    return input(f"{prompt} (y/n): ").strip().lower() == "y"


# ─────────────────────────────────────────────
# Database Setup
# ─────────────────────────────────────────────
def setup_database():
    utils.clear_screen()
    console.print("[bold cyan]--- Setting Up Database ---[/bold cyan]")
    if not confirm_continue("This will DROP existing tables. Continue?"):
        return

    stage3_etl.create_tables()
    stage4_global_analysis.setup_database_tables()

    console.print("[green]Database setup complete.[/green]")
    input("\nPress Enter to return...")


# ─────────────────────────────────────────────
# Stage Launchers
# ─────────────────────────────────────────────
def run_stage_1():
    utils.clear_screen()
    stage1_generate_users.main()
    input("\nPress Enter to return...")


def run_stage_2():
    utils.clear_screen()
    ok, msg = utils.check_file_prerequisites(2)
    if not ok:
        console.print(f"[red]{msg}[/red]")
        if confirm_continue("Run Stage 1 now?"):
            run_stage_1()
            console.print("Retrying Stage 2...")
            stage2_generate_posts.main()
    else:
        stage2_generate_posts.main()
    input("\nPress Enter to return...")


def run_stage_3():
    utils.clear_screen()
    ok, msg = utils.check_file_prerequisites(3)
    if not ok:
        console.print(f"[red]{msg}[/red]")
    else:
        stage3_etl.run_etl()
    input("\nPress Enter to return...")


def run_stage_4():
    utils.clear_screen()
    console.print("[bold cyan]--- Launching Batch Topic Analysis ---[/bold cyan]")
    if confirm_continue("Continue?"):
        stage4_global_analysis.run_global_analysis()
    input("\nPress Enter to return...")


def run_stage_5():
    utils.clear_screen()
    stage5_ui_helpers.view_global_topics()
    input("\nPress Enter to return...")


# ─────────────────────────────────────────────
# NEW: Live Analysis Interface
# ─────────────────────────────────────────────
def run_live_analysis_menu():
    """
    Presents source selection:
      • Reddit
      • LinkedIn
    Then calls the unified Stage 4 live-analysis pipeline.
    """

    while True:
        utils.clear_screen()
        console.rule("[bold magenta]Live Social Feed Analysis[/bold magenta]")

        console.print("[cyan]1.[/cyan] Analyze Reddit Feed")
        console.print("[cyan]2.[/cyan] Analyze LinkedIn Feed")
        console.print("[cyan]3.[/cyan] Back")
        console.print("-" * 50)

        choice = input("Choose source: ").strip()

        if choice == "3":
            return

        if choice not in ("1", "2"):
            input("Invalid option. Press Enter...")
            continue

        # Choose user
        utils.clear_screen()
        stage5_ui_helpers.list_all_users()

        user_id = input("\nEnter User ID to analyze (e.g., user_1): ").strip()
        if not user_id:
            input("Invalid user. Press Enter...")
            continue

        # Choose max items
        try:
            top_n = int(input("How many posts to process (recommended 50–100)? ").strip())
        except ValueError:
            top_n = 50

        utils.clear_screen()
        console.print("[cyan]Running live topic analysis...[/cyan]")

        # Select source
        if choice == "1":
            # Reddit
            stage4_global_analysis.run_live_analysis_for_user(
                user_id=user_id,
                source="reddit",
                fetcher_callable=fetch_reddit_user_posts,
                top_n=top_n,
            )
        elif choice == "2":
            # LinkedIn
            stage4_global_analysis.run_live_analysis_for_user(
                user_id=user_id,
                source="linkedin",
                fetcher_callable=fetch_linkedin_user_posts,
                top_n=top_n,
            )

        input("\nPress Enter to return...")


# ─────────────────────────────────────────────
# Main Menu
# ─────────────────────────────────────────────
def main_menu():
    options = {
        "1": ("Setup Database", setup_database),
        "2": ("Run Stage 1: Generate Users", run_stage_1),
        "3": ("Run Stage 2: Generate Posts", run_stage_2),
        "4": ("Run Stage 3: Run ETL", run_stage_3),
        "5": ("Run Stage 4: Batch Topic Analysis", run_stage_4),
        "6": ("Run Stage 5: View Global Topics", run_stage_5),
        "7": ("List All Users", stage5_ui_helpers.list_all_users),
        "8": ("Live Social Feed Analysis", run_live_analysis_menu),
        "9": ("Exit", lambda: sys.exit(console.print("[bold red]Exiting...[/bold red]"))),
    }

    while True:
        utils.clear_screen()
        console.rule("[bold magenta]Social Media Topic & Summary Tool[/bold magenta]")

        for key, (label, _) in options.items():
            console.print(f"[cyan]{key}.[/cyan] {label}")

        console.print("-" * 50)
        choice = input("Enter choice: ").strip()

        if choice in options:
            utils.clear_screen()
            _, action = options[choice]
            action()
        else:
            input("Invalid option. Press Enter...")


# ─────────────────────────────────────────────
# Click CLI
# ─────────────────────────────────────────────
@click.group()
def cli():
    pass


@cli.command()
def stage4():
    console.print("[cyan]Running Stage 4 Analysis[/cyan]")
    stage4_global_analysis.run_global_analysis()


@cli.command()
def live():
    console.print("[cyan]Launching Live Analysis Menu[/cyan]")
    run_live_analysis_menu()


@cli.command()
def menu():
    main_menu()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli()
    else:
        main_menu()
