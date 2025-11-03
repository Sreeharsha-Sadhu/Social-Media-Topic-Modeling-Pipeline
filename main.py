# main.py

import sys
import utils
import _1_generation
import _2_content
import _3_etl
import _4_analysis
import _5_ui


def setup_database():
    """Runs all table creation scripts."""
    utils.clear_screen()
    print("--- 🛠️ Setting Up Database ---")
    _3_etl.create_tables()
    _4_analysis.create_results_tables()
    print("\nDatabase setup complete.")
    input("\nPress Enter to return to the menu...")


def run__1():
    """Runs Stage 1: Data Generation"""
    utils.clear_screen()
    _1_generation.main()
    input("\nPress Enter to return to the menu...")


def run__2():
    """Runs Stage 2: Content Generation"""
    utils.clear_screen()
    success, msg = utils.check_file_prerequisites(2)
    if not success:
        print(f"🚨 Prerequisite Failed: {msg}")
        if input("Run Stage 1 now? (y/n): ").lower() == 'y':
            run__1()
            utils.clear_screen()
            print("Retrying Stage 2...")
            _2_content.main()
        else:
            print("Aborting Stage 2.")
    else:
        _2_content.main()
    input("\nPress Enter to return to the menu...")


def run__3():
    """Runs Stage 3: ETL Pipeline"""
    utils.clear_screen()
    success, msg = utils.check_file_prerequisites(3)
    if not success:
        print(f"🚨 Prerequisite Failed: {msg}")
        print("Please run Stages 1 and 2 before running the ETL.")
    else:
        _3_etl.run_etl()
    input("\nPress Enter to return to the menu...")


def run__4():
    """Runs Stage 4: AI Analysis"""
    utils.clear_screen()
    try:
        _5_ui.list_all_users()
    except Exception:
        print("--- 🚨 WARNING ---")
        print("Could not list users. The database tables might be empty.")
        if input("Run Stage 3 (ETL) first? (y/n): ").lower() != 'y':
            return
        run__3()
        utils.clear_screen()
        _5_ui.list_all_users()
    
    user_id = input("\nEnter User ID to analyze (e.g., user_1): ")
    if not user_id:
        print("Invalid User ID.")
        return
    
    print(f"\nStarting new analysis for {user_id}...")
    _4_analysis.run_analysis_pipeline(user_id)
    print(f"\n✅ Analysis complete for {user_id}!")
    input("\nPress Enter to return to the menu...")


def run__5():
    """Runs Stage 5: View Results"""
    utils.clear_screen()
    user_id = input("Enter User ID to view (e.g., user_1): ")
    if not user_id:
        print("Invalid User ID.")
        return
    _5_ui.view_results_for_user(user_id)
    input("\nPress Enter to return to the menu...")


def main_menu():
    """The main interactive console loop."""
    menu_options = {
        "1": {"text": "Setup Database (Run First)", "action": setup_database},
        "2": {"text": "Run Stage 1: Generate Users & Graph", "action": run__1},
        "3": {"text": "Run Stage 2: Generate Post Content", "action": run__2},
        "4": {"text": "Run Stage 3: Run ETL Pipeline (PySpark)", "action": run__3},
        "5": {"text": "Run Stage 4: Run AI Analysis on a User", "action": run__4},
        "6": {"text": "Run Stage 5: View User Analysis Results", "action": run__5},
        "7": {"text": "Exit", "action": lambda: sys.exit("Exiting. Goodbye!")}
    }
    
    while True:
        utils.clear_screen()
        print("=" * 50)
        print("      📊 Social Media Topic & Summary Tool 📊")
        print("=" * 50)
        for key, value in menu_options.items():
            print(f"{key}. {value['text']}")
        print("-" * 50)
        
        choice = input("Enter your choice: ")
        
        if choice in menu_options:
            menu_options[choice]["action"]()
        else:
            input("Invalid choice. Press Enter to try again...")


if __name__ == "__main__":
    main_menu()
