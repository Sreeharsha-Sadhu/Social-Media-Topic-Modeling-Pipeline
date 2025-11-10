# Social Media Topic & Summary Analysis Tool

A modular pipeline for generating, processing, and analyzing synthetic social media data using both **Pandas** and **PySpark**.  
This project performs complete end-to-end topic modeling, clustering, and summarization with optional GPU acceleration.

---

## Overview

The system simulates social media interactions, processes them through an ETL pipeline, and applies AI-driven topic analysis using transformer-based summarization.  
It supports both **menu-based** and **command-line (Click)** interfaces for flexible use.

---

## Features

- Multi-stage data pipeline:
  1. Synthetic user and network generation
  2. Post content creation
  3. Dual-mode ETL (Pandas / PySpark)
  4. AI-based topic modeling and summarization
  5. Interactive and CLI-based visualization
- PostgreSQL backend for persistence
- Optional Spark acceleration for ETL and clustering
- Extensible for live social feed analysis (Reddit, Twitter, LinkedIn)
- Ready for future Streamlit web UI integration

---

## Prerequisites

- **Python** ≥ 3.10  
- **Java JDK** ≥ 17  
- **PostgreSQL** (local or remote)
- **Apache Spark** ≥ 3.5 (tested up to 4.0.1)
- **Virtual environment** (recommended)

Verify your Spark setup using:

```bash
python tools/spark_env_check.py
````

A successful setup will confirm Spark version, Java linkage, and Python worker health.

---

## Project Structure

```
src/
├── core/              # Configuration, utils, logging
├── pipeline/          # ETL and analysis stages (1–4)
├── ui/                # CLI + future Streamlit UI
├── analysis/          # Live feed analyzers (future)
├── __init__.py
data/                  # Generated JSON and graph files
logs/                  # Application logs
tools/                 # Diagnostic utilities
tests/                 # Unit tests
```

---

## Environment Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/<username>/social-media-analysis.git
   cd social-media-analysis
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate     # Linux/macOS
   .venv\Scripts\activate        # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

5. Edit `.env` with your database and Spark settings:

   ```bash
   DB_NAME=your_db_name
   DB_USER=your_db_user
   DB_PASS=your_db_password
   DB_HOST=localhost
   DB_PORT=5432
   USE_SPARK_ETL=true
   SPARK_MASTER=local[*]
   SPARK_APP_NAME=SocialMediaETL
   ```

---

## Running the Application

### Option 1: Interactive Menu

```bash
python src/ui/cli_main.py
```

### Option 2: Command-Line Mode (Click Interface)

```bash
python src/ui/cli_main.py stage1
python src/ui/cli_main.py stage2
python src/ui/cli_main.py stage3 --mode spark
python src/ui/cli_main.py stage4
python src/ui/cli_main.py stage5
```

---

## Verifying Spark Installation

Run the diagnostic tool to verify PySpark configuration:

```bash
python tools/spark_env_check.py
```

Expected output includes:

* Platform and Python version details
* Verified Spark and Java versions
* Confirmation that Python workers are alive and consistent

If successful, you should see:

```
✅ Spark Python linkage OK — all workers alive and consistent!
```

---

## Database Schema Overview

| Table                | Description                               |
| -------------------- | ----------------------------------------- |
| `users`              | Stores user IDs, personas, and subreddits |
| `follows`            | Directed follow relationships             |
| `posts`              | Raw and cleaned user posts                |
| `global_topics`      | AI-generated topic summaries              |
| `post_topic_mapping` | Mapping of posts to topics                |

---

## Logs and Diagnostics

* Logs are written to `logs/app.log` by default.
* Rich console output is enabled for real-time progress.
* For detailed Spark logs, refer to `$SPARK_HOME/logs`.

---

## License

This project is for academic and research purposes.
Use of external API credentials (e.g., Reddit, Twitter) requires proper authorization and compliance with respective platform terms.

---

## Notes

* All confidential environment variables (database, API keys) must remain in `.env` and **should never be committed**.
* The `tools/spark_env_check.py` script can be used to validate Spark-Python integration after every environment setup.
* Live analysis modules are placeholders and will be implemented after completion of the batch pipeline.

```
```
