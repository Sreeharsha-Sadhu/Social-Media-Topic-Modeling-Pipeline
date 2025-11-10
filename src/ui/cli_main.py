"""
CLI and Menu Entry Point
──────────────────────────────────────────────
Provides two interfaces for running the Social Media Topic & Summary Tool:
  1. Menu-based UI (interactive, for exploration)
  2. Click-based CLI (automated, for scripts & pipelines)
"""

import sys
import logging
import click
from rich.console import Console
from rich.logging import RichHandler

from src.core import utils
from src.data_pipeline import stage1_generate_users, stage2_generate_posts, stage3_etl, stage4_global_analysis
from src.ui import stage5_ui_helpers

# Optional live analysis imports (currently on hold)
try:
    from src.analysis.live import analyze_reddit_feed, analyze_twitter_feed, analyze_linkedin_feed
    LIVE_ANALYSIS_AVAILABLE = True
except ImportError:
    LIVE_ANALYSIS_AVAILABLE = False

# ─────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True)]
)
logger = logging.getLogger("CLI")

# ─────────────────────────────────────────────
# Shared Utility Functions
# ─────────────────────────────────────────────
def confirm_continue(prompt: str) -> bool:
    """Prompt user for yes/no confirmation."""
    return input(f"{prompt} (y/n): ").strip().lower() == "y"


def setup_database():
    """Create all required tables."""
    utils.clear_screen()
    console.print("[bold cyan]--- 🛠️ Setting Up Database ---[/bold cyan]")
    if not confirm_continue("⚠️ This will DROP existing tables. Continue?"):
        return
    stage3_etl.create_tables()
    stage4_global_analysis.setup_database_tables()
    console.print("[green]✅ Database setup complete.[/green]")
    input("Press Enter to return...")


# ─────────────────────────────────────────────
# Menu-Based Interface
# ─────────────────────────────────────────────
def run_stage_1():
    utils.clear_screen()
    stage1_generate_users.main()
    input("\nPress Enter to return...")


def run_stage_2():
    utils.clear_screen()
    success, msg = utils.check_file_prerequisites(2)
    if not success:
        console.print(f"[red]🚨 {msg}[/red]")
        if confirm_continue("Run Stage 1 now?"):
            run_stage_1()
            console.print("[green]Retrying Stage 2...[/green]")
            stage2_generate_posts.main()
    else:
        stage2_generate_posts.main()
    input("\nPress Enter to return...")


def run_stage_3():
    utils.clear_screen()
    success, msg = utils.check_file_prerequisites(3)
    if not success:
        console.print(f"[red]🚨 {msg}[/red]")
    else:
        stage3_etl.run_etl()
    input("\nPress Enter to return...")


def run_stage_4():
    utils.clear_screen()
    console.print("[bold cyan]--- 🚀 Launching Global Topic Model ---[/bold cyan]")
    if confirm_continue("Continue with AI-based topic analysis?"):
        stage4_global_analysis.run_global_analysis()
    input("\nPress Enter to return...")


def run_stage_5():
    utils.clear_screen()
    stage5_ui_helpers.view_global_topics()
    input("\nPress Enter to return...")


def main_menu():
    """Interactive numbered console interface."""
    options = {
        "1": ("Setup Database (Run First!)", setup_database),
        "2": ("Run Stage 1: Generate Users & Graph", run_stage_1),
        "3": ("Run Stage 2: Generate Post Content", run_stage_2),
        "4": ("Run Stage 3: Run ETL Pipeline", run_stage_3),
        "5": ("Run Stage 4: Generate Global Topic Model (Batch AI)", run_stage_4),
        "6": ("Run Stage 5: View Global Topic Results", run_stage_5),
        "7": ("List All Users", stage5_ui_helpers.list_all_users),
        "9": ("Exit", lambda: sys.exit(console.print("[bold red]Exiting... Goodbye![/bold red]"))),
    }

    while True:
        utils.clear_screen()
        console.rule("[bold magenta]📊 Social Media Topic & Summary Tool[/bold magenta]")
        for key, (label, _) in options.items():
            console.print(f"[cyan]{key}[/cyan]. {label}")
        console.print("-" * 50)
        choice = input("Enter your choice: ").strip().lower()

        if choice in options:
            utils.clear_screen()
            _, action = options[choice]
            action()
        else:
            input("Invalid choice. Press Enter to retry...")


# ─────────────────────────────────────────────
# Click-Based Command-Line Interface
# ─────────────────────────────────────────────
@click.group()
def cli():
    """Social Media Topic & Summary Tool (Command-Line Interface)."""
    pass


@cli.command()
@click.option("--mode", type=click.Choice(["pandas", "spark"]), default="pandas", help="Select ETL mode.")
def stage3(mode):
    """Run ETL (Stage 3)."""
    console.print(f"[cyan]Running ETL in {mode.upper()} mode...[/cyan]")
    if mode == "spark":
        stage3_etl.run_etl()
    else:
        stage3_etl._run_etl_pandas()


@cli.command()
def stage4():
    """Run AI-based Global Topic Analysis (Stage 4)."""
    console.print("[cyan]Running Global Topic Model (AI)...[/cyan]")
    stage4_global_analysis.run_global_analysis()


@cli.command()
def stage5():
    """View summarized topics (Stage 5)."""
    console.print("[cyan]Viewing Global Topics...[/cyan]")
    stage5_ui_helpers.view_global_topics()


@cli.command()
def users():
    """List all users."""
    stage5_ui_helpers.list_all_users()


@cli.command()
def menu():
    """Launch interactive menu interface."""
    main_menu()


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli()  # Run click mode
    else:
        main_menu()  # Run interactive mode
