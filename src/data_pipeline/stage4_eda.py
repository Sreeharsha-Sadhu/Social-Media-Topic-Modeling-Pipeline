"""
stage4_eda.py

Exploratory Data Analysis helpers (Spark-based) used by stage4_global_analysis.
Provides:
 - run_global_eda_from_rows(spark, rows): returns basic EDA stats (term frequency, length, sentiment histogram)
 - utility helpers for TF, TF-IDF (Spark ML)
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from src.core.logging_config import get_logger

logger = get_logger(__name__)

def run_global_eda_from_rows(spark: SparkSession, rows: List[Tuple[str, str, str, str, int, int]]) -> Dict[str, Any]:
    """
    rows: list of tuples (post_id, subreddit, content, created_at, score, num_comments)
    Returns:
      {
        'overall_post_count': int,
        'posts_by_subreddit': {sub: count},
        'avg_post_length': float,
        'top_terms': [(term, count), ...],
        'score_stats': {avg, median, stddev},
        'comments_stats': {avg, median, stddev}
      }
    """
    if not rows:
        return {}

    schema = T.StructType([
        T.StructField("post_id", T.StringType(), True),
        T.StructField("subreddit", T.StringType(), True),
        T.StructField("content", T.StringType(), True),
        T.StructField("created_at", T.StringType(), True),
        T.StructField("score", T.IntegerType(), True),
        T.StructField("num_comments", T.IntegerType(), True),
    ])
    df = spark.createDataFrame(rows, schema=schema)

    # basic counts
    overall_post_count = df.count()
    posts_by_sub = {row["subreddit"]: int(row["cnt"]) for row in df.groupBy("subreddit").count().collect()}

    # length stats
    df = df.withColumn("char_len", F.length(F.col("content")))
    avg_len = float(df.agg(F.avg(F.col("char_len"))).first()[0] or 0.0)

    # score/comments stats
    score_stats_row = df.select(F.avg("score").alias("avg"), F.expr("percentile_approx(score, 0.5)").alias("median"),
                                F.stddev("score").alias("stddev")).first()
    comments_stats_row = df.select(F.avg("num_comments").alias("avg"), F.expr("percentile_approx(num_comments, 0.5)").alias("median"),
                                   F.stddev("num_comments").alias("stddev")).first()
    score_stats = {"avg": float(score_stats_row["avg"] or 0.0), "median": int(score_stats_row["median"] or 0), "stddev": float(score_stats_row["stddev"] or 0.0)}
    comments_stats = {"avg": float(comments_stats_row["avg"] or 0.0), "median": int(comments_stats_row["median"] or 0), "stddev": float(comments_stats_row["stddev"] or 0.0)}

    # tokenization + stopword removal + top terms (HashingTF -> IDF expensive; use HashingTF counts)
    tokenizer = Tokenizer(inputCol="content", outputCol="tokens_raw")
    df_tokens = tokenizer.transform(df)
    swr = StopWordsRemover(inputCol="tokens_raw", outputCol="tokens_filtered")
    df_tokens = swr.transform(df_tokens)
    hashingTF = HashingTF(inputCol="tokens_filtered", outputCol="tf", numFeatures=1 << 18)
    df_tf = hashingTF.transform(df_tokens)

    # For top terms manually explode tokens_filtered and count (small, OK for EDA)
    df_exploded = df_tokens.select(F.explode(F.col("tokens_filtered")).alias("token"))
    df_terms = df_exploded.groupBy("token").count().orderBy(F.desc("count")).limit(40)
    top_terms = [(row["token"], int(row["count"])) for row in df_terms.collect()]

    result = {
        "overall_post_count": int(overall_post_count),
        "posts_by_subreddit": posts_by_sub,
        "avg_post_length_chars": avg_len,
        "top_terms": top_terms,
        "score_stats": score_stats,
        "comments_stats": comments_stats
    }
    return result
