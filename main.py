# main.py

import sys
import utils
import stage_1_generation
import stage_2_content
import stage_3_etl
import stage_4_analysis
import stage_5_ui_helpers
import live_analyzer  # <-- ADDED

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
        print(f"--- 🚨 A critical error occurred during analysis ---")
        print(f"Error: {e}")
    input("\nPress Enter to return to the menu...")


def run_stage_5():
    utils.clear_screen()
    stage_5_ui_helpers.view_global_topics()
    input("\nPress Enter to return to the menu...")


# --- NEW: Function to run the live analysis ---
def run_live_analysis():
    utils.clear_screen()
    print("This will fetch live data from the Reddit API.")
    print("It requires a working internet connection and valid API keys in config.py")
    
    stage_5_ui_helpers.list_all_users()
    user_id = input("\nEnter User ID to analyze (e.g., user_1): ")
    if not user_id:
        print("Invalid User ID.")
        input("\nPress Enter to return to the menu...")
        return
    
    live_analyzer.analyze_live_feed(user_id)
    input("\nPress Enter to return to the menu...")


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
        "7": {"text": "Analyze Live Reddit Feed (New!)", "action": run_live_analysis},  # <-- ADDED
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
