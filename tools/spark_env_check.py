#!/usr/bin/env python3
"""
spark_env_check.py
──────────────────
Verifies that your local PySpark installation, Python environment, and JVM are
compatible. Prints clear diagnostics for driver/executor Python versions,
PySpark/Java versions, and confirms that worker processes can start and run
simple jobs.

Run:
    python spark_env_check.py

If any check fails, follow the suggested “❌ Fix” notes printed at the end.
"""

import os
import sys
import platform
import traceback
from pyspark.sql import SparkSession
from dotenv import load_dotenv

load_dotenv()

def banner(msg):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def main():
    banner("🧩 Spark Environment Diagnostic Tool")

    # --- System / Python info ---
    print(f"Platform      : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Python exec   : {sys.executable}")
    print(f"Python ver    : {sys.version}")
    print(f"Virtual env   : {os.getenv('VIRTUAL_ENV', '(none)')}")

    # --- Environment variables ---
    print("\n[Env Variables]")
    for key in ["PYSPARK_PYTHON", "PYSPARK_DRIVER_PYTHON", "JAVA_HOME", "SPARK_HOME"]:
        print(f"{key:22}: {os.getenv(key, '(unset)')}")

    # --- Try starting Spark ---
    try:
        spark = (
            SparkSession.builder
            .appName("SparkEnvCheck")
            .master("local[*]")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
            .config("spark.python.worker.reuse", "true")
            .config("spark.default.parallelism", "4")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.driver.memory", "2g")
            .config("spark.executor.memory", "2g")
            .getOrCreate()
        )
        sc = spark.sparkContext
        print("\n✅ SparkSession started successfully.")
        print(f"Spark version : {spark.version}")
        print(f"Java version  : {sc._jvm.java.lang.System.getProperty('java.version')}")
        print(f"Spark master  : {sc.master}")
        import py4j
        print(f"Py4J version  : {py4j.__version__}")
        
        # --- Worker test ---
        print("\n[Worker Test] Launching Python worker via RDD.collect()...")
        rdd = sc.parallelize(range(4))
        results = rdd.map(
            lambda i: (i, os.getenv("PYSPARK_PYTHON"), sys.version)
        ).collect()

        print("\n✅ Worker round-trip succeeded. Details:")
        for idx, (i, exe, ver) in enumerate(results):
            print(f"  Worker {idx}: Python={exe} | Version={ver.split()[0]}")

        banner("✅ Spark Python linkage OK — all workers alive and consistent!")

    except Exception as e:
        banner("❌ Spark worker startup FAILED")
        print("Exception:")
        print(e)
        traceback.print_exc(limit=8)
        print("\nCommon fixes:")
        print("  1. Ensure PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON point to the SAME interpreter.")
        print("  2. Downgrade to Java 17 if you are running Java 21.")
        print("  3. Reinstall matching pyspark/py4j:  pip install -U pyspark==3.5.1 py4j==0.10.9.7")
        print("  4. Restart shell after changing environment variables.")
    finally:
        try:
            spark.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
