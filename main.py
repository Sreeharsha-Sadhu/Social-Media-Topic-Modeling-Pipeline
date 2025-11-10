# main.py

import sys
from src.common import utils
from src.etl import stage_3_etl, stage_1_generation, stage_2_content
from src.analysis import stage_4_analysis
from src.ui import stage_5_ui_helpers
# --- MODIFIED IMPORTS ---
from src.analysis.live_analyzer import (
    analyze_reddit_feed,
    analyze_linkedin_feed,
    analyze_twitter_feed
)

# ------------------------
try:
    import stage_3_etl_spark_demo
    
    SPARK_DEMO_AVAILABLE = True
except ImportError:
    SPARK_DEMO_AVAILABLE = False


def setup_database():
    utils.clear_screen()
    print("--- 🛠️ Setting Up Database ---")
    print("This will create tables for the core Pandas pipeline.")
    print("WARNING: This will DROP 'global_topics' and 'post_topic_mapping' tables if they exist.")
    if input("Continue? (y/n): ").lower() != 'y':
        return
    
    stage_3_etl.create_tables()
    stage_4_analysis.setup_database_tables()
    
    print("\nDatabase setup complete.")
    input("\nPress Enter to return to the menu...")


# (stage_1, 2, 3, 4, 5 functions are unchanged)
def run_stage_1():
    utils.clear_screen()
    stage_1_generation.main()
    input("\nPress Enter to return to the menu...")


def run_stage_2():
    utils.clear_screen()
    success, msg = utils.check_file_prerequisites(2)
    if not success:
        print(f"🚨 Prerequisite Failed: {msg}")
        if input("Run Stage 1 now? (y/n): ").lower() == 'y':
            run_stage_1()
            utils.clear_screen()
            print("Retrying Stage 2...")
            stage_2_content.main()
        else:
            print("Aborting Stage 2.")
    else:
        stage_2_content.main()
    input("\nPress Enter to return to the menu...")


def run_stage_3():
    utils.clear_screen()
    success, msg = utils.check_file_prerequisites(3)
    if not success:
        print(f"🚨 Prerequisite Failed: {msg}")
        print("Please run Stages 1 and 2 before running the ETL.")
    else:
        stage_3_etl.run_etl()
    input("\nPress Enter to return to the menu...")


def run_stage_4():
    utils.clear_screen()
    print("--- 🚀 Launching Stage 4: Generate Global Topic Model (Pandas/AI) ---")
    print("This will use your CPU/GPU (if available) to analyze all posts in the database.")
    if input("Continue? (y/n): ").lower() != 'y':
        return
    try:
        stage_4_analysis.run_global_analysis()
    except Exception as e:
        print(f"--- 🚨 A critical error occurred during analysis --- \nError: {e}")
    input("\nPress Enter to return to the menu...")


def run_stage_5():
    utils.clear_screen()
    stage_5_ui_helpers.view_global_topics()
    input("\nPress Enter to return to the menu...")


# --- MODIFIED: This is now a sub-menu ---
def run_live_analysis_menu():
    """Shows a sub-menu for choosing a live data source."""
    while True:
        utils.clear_screen()
        print("--- 📡 Analyze Live Feed ---")
        print("1. Analyze Reddit Feed (using user's subreddits)")
        print("2. Analyze Twitter/X Feed (requires login)")
        print("3. Analyze LinkedIn Feed (requires login)")
        print("4. Back to Main Menu")
        print("-" * 50)
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            utils.clear_screen()
            stage_5_ui_helpers.list_all_users()
            user_id = input("\nEnter User ID to analyze (e.g., user_1): ")
            if user_id:
                analyze_reddit_feed(user_id)
            input("\nPress Enter to return...")
        
        elif choice == '2':
            utils.clear_screen()
            print("--- Twitter/X Login ---")
            print("Your credentials are NOT saved and are used only for this session.")
            username = input("Enter Twitter/X Email or Username: ")
            password = input("Enter Twitter/X Password: ")
            if username and password:
                analyze_twitter_feed(username, password)
            input("\nPress Enter to return...")
        
        elif choice == '3':
            utils.clear_screen()
            print("--- LinkedIn Login ---")
            print("Your credentials are NOT saved and are used only for this session.")
            username = input("Enter LinkedIn Email: ")
            password = input("Enter LinkedIn Password: ")  # Hides password
            if username and password:
                analyze_linkedin_feed(username, password)
            input("\nPress Enter to return...")
        
        elif choice == '4':
            break
        else:
            input("Invalid choice. Press Enter to try again...")


def run_spark_demo():
    utils.clear_screen()
    if not SPARK_DEMO_AVAILABLE:
        print("--- 🚨 ERROR ---")
        print("Could not find 'stage_3_etl_spark_demo.py' or 'pyspark'.")
        print("Please ensure you have run: pip install -r requirements_spark_demo.txt")
        input("\nPress Enter to return to the menu...")
        return
    
    print("--- 🚀 Launching Stage 3: PySpark ETL Demo ---")
    if input("Continue? (y/n): ").lower() != 'y':
        return
    
    stage_3_etl_spark_demo.run_etl()
    input("\nPress Enter to return to the menu...")


def main_menu():
    """The main interactive console loop."""
    menu_options = {
        "1": {"text": "Setup Database (Run First!)", "action": setup_database},
        "2": {"text": "Run Stage 1: Generate Users & Graph", "action": run_stage_1},
        "3": {"text": "Run Stage 2: Generate Post Content", "action": run_stage_2},
        "4": {"text": "Run Stage 3: Run ETL Pipeline (Pandas)", "action": run_stage_3},
        "5": {"text": "Run Stage 4: Generate Global Topic Model (Batch AI)", "action": run_stage_4},
        "6": {"text": "Run Stage 5: View Global Topic Results", "action": run_stage_5},
        "7": {"text": "Analyze Live Feed (Reddit, Twitter, LinkedIn)", "action": run_live_analysis_menu},
        "8": {"text": "List All Users", "action": stage_5_ui_helpers.list_all_users},
        "9": {"text": "Exit", "action": lambda: sys.exit("Exiting. Goodbye!")}
    }
    
    if SPARK_DEMO_AVAILABLE:
        menu_options["d"] = {"text": "Run Stage 3 (PySpark Demo)", "action": run_spark_demo}
    
    while True:
        utils.clear_screen()
        print("=" * 50)
        print("      📊 Social Media Topic & Summary Tool 📊")
        print("=" * 50)
        for key, value in menu_options.items():
            print(f"{key}. {value['text']}")
        print("-" * 50)
        
        choice = input("Enter your choice: ").lower()
        
        if choice in menu_options:
            utils.clear_screen()
            menu_options[choice]["action"]()
        else:
            input("Invalid choice. Press Enter to try again...")


if __name__ == "__main__":
    main_menu()
