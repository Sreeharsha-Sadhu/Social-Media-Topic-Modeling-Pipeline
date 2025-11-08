# main.py

import sys
import utils
import _1_generation
import _2_content
import _3_etl  # <-- The new Pandas ETL
import _4_analysis  # <-- The new Pandas AI
import _5_ui

# --- NEW: Import the demo script ---
try:
    import _3_etl_spark_demo
    
    SPARK_DEMO_AVAILABLE = True
except ImportError:
    SPARK_DEMO_AVAILABLE = False


def setup_database():
    """Runs all table creation scripts."""
    utils.clear_screen()
    print("--- 🛠️ Setting Up Database ---")
    print("This will create tables for the core Pandas pipeline.")
    print("WARNING: This will DROP 'global_topics' and 'post_topic_mapping' tables if they exist.")
    if input("Continue? (y/n): ").lower() != 'y':
        return
    
    _3_etl.create_tables()  # Creates users, posts, follows
    _4_analysis.setup_database_tables()  # Creates global_topics, post_topic_mapping
    
    print("\nDatabase setup complete.")
    input("\nPress Enter to return to the menu...")


# (run_stage_1 and run_stage_2 are unchanged)
def run_stage_1():
    utils.clear_screen()
    _1_generation.main()
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
            _2_content.main()
        else:
            print("Aborting Stage 2.")
    else:
        _2_content.main()
    input("\nPress Enter to return to the menu...")


# --- MODIFIED: Calls the new Pandas ETL ---
def run_stage_3():
    """Runs Stage 3: Pandas ETL Pipeline"""
    utils.clear_screen()
    success, msg = utils.check_file_prerequisites(3)
    if not success:
        print(f"🚨 Prerequisite Failed: {msg}")
        print("Please run Stages 1 and 2 before running the ETL.")
    else:
        _3_etl.run_etl()  # Calls the Pandas version
    input("\nPress Enter to return to the menu...")


# --- MODIFIED: Calls the new Pandas AI ---
def run_stage_4():
    """Runs Stage 4: Global Pandas Analysis"""
    utils.clear_screen()
    print("--- 🚀 Launching Stage 4: Global Pandas/SKlearn Analysis ---")
    print("This will use your CPU/GPU (if available) to analyze all posts.")
    print("This may take several minutes.")
    if input("Continue? (y/n): ").lower() != 'y':
        return
    
    try:
        _4_analysis.run_global_analysis()
    except Exception as e:
        print(f"--- 🚨 A critical error occurred during analysis ---")
        print(f"Error: {e}")
    
    input("\nPress Enter to return to the menu...")


# --- MODIFIED: Calls the renamed helper ---
def run_stage_5():
    """Runs Stage 5: View Global Topic Results"""
    utils.clear_screen()
    _5_ui.view_global_topics()
    input("\nPress Enter to return to the menu...")


# --- NEW: Function to run the Spark demo ---
def run_spark_demo():
    """Runs the optional Stage 3 Spark Demo"""
    utils.clear_screen()
    if not SPARK_DEMO_AVAILABLE:
        print("--- 🚨 ERROR ---")
        print("Could not find '_3_etl_spark_demo.py' or 'pyspark'.")
        print("Please ensure you have run: pip install -r requirements_spark_demo.txt")
        input("\nPress Enter to return to the menu...")
        return
    
    print("--- 🚀 Launching Stage 3: PySpark ETL Demo ---")
    print("This will run the *exact same* ETL logic, but using PySpark.")
    if input("Continue? (y/n): ").lower() != 'y':
        return
    
    _3_etl_spark_demo.run_etl()
    input("\nPress Enter to return to the menu...")


def main_menu():
    """The main interactive console loop."""
    menu_options = {
        "1": {"text": "Setup Database (Run First!)", "action": setup_database},
        "2": {"text": "Run Stage 1: Generate Users & Graph", "action": run_stage_1},
        "3": {"text": "Run Stage 2: Generate Post Content", "action": run_stage_2},
        "4": {"text": "Run Stage 3: Run ETL Pipeline (Pandas)", "action": run_stage_3},
        "5": {"text": "Run Stage 4: Generate Global Topic Model (Pandas/AI)", "action": run_stage_4},
        "6": {"text": "Run Stage 5: View Global Topic Results", "action": run_stage_5},
        "7": {"text": "List All Users", "action": _5_ui.list_all_users},
        "8": {"text": "Exit", "action": lambda: sys.exit("Exiting. Goodbye!")}
    }
    
    # Dynamically add the Spark demo if it's available
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
