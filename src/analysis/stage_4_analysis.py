# stage_4_analysis.py

import numpy as np
import pandas as pd
import torch
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql import functions as F
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans as SklearnKMeans
from tqdm import tqdm
from transformers import pipeline

from src.config import settings
from src.common import utils
from src.common.utils import get_spark_session

# (get_llm_pipeline function is unchanged)
model_cache = {}


def get_llm_pipeline():
    if "llm" not in model_cache:
        print("--- Caching: Loading FLAN-T5-Large model (770M params) to GPU with FP16 ---")
        print("   (This may take several minutes and >3GB of VRAM on first run)")
        model_name = "google/flan-t5-large"
        
        if not torch.cuda.is_available():
            print("--- WARNING: CUDA not available. Loading model on CPU. This will be VERY slow. ---")
            model_cache["llm"] = pipeline("text2text-generation", model=model_name)
        else:
            model_cache["llm"] = pipeline(
                "text2text-generation",
                model=model_name,
                device=0,
                dtype=torch.float16  # Use correct 'dtype'
            )
    return model_cache["llm"]


def run_global_analysis():
    print("--- Starting Stage 4: Global Analysis ---")
    mode = "Spark" if settings.USE_SPARK_ANALYSIS else "Pandas"
    print(f"Analysis Mode: {mode}")
    
    try:
        engine = utils.get_sqlalchemy_engine()
    except Exception as e:
        print(f"🚨 Could not create SQLAlchemy engine: {e}")
        return False
    
    if settings.USE_SPARK_ANALYSIS:
        return _run_global_analysis_spark(engine)
    else:
        return _run_global_analysis_pandas(engine)


def _run_global_analysis_pandas(engine):
    try:
        print("Loading all posts from PostgreSQL into Pandas...")
        sql_query = "SELECT post_id, cleaned_content FROM posts WHERE cleaned_content IS NOT NULL AND cleaned_content != ''"
        pd_posts = pd.read_sql(sql_query, engine)
        
        if len(pd_posts) < 20:
            print(f"Not enough posts ({len(pd_posts)}) to analyze. Aborting.")
            return False
        print(f"Loaded {len(pd_posts)} posts.")
        
        print("Loading sentence-transformer model (in-driver)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        print("Generating embeddings (in-driver)...")
        embeddings = sbert_model.encode(
            pd_posts['cleaned_content'].tolist(),
            batch_size=16,
            show_progress_bar=True
        )
        
        NUM_TOPICS = 20
        print(f"Clustering posts into {NUM_TOPICS} topics with scikit-learn K-Means...")
        kmeans = SklearnKMeans(n_clusters=NUM_TOPICS, random_state=0, n_init=10)
        pd_posts['topic_id'] = kmeans.fit_predict(embeddings)
        
        print("Summarizing topics (in-driver with FLAN-T5-Large)...")
        generator = get_llm_pipeline()
        
        topic_summaries = []
        
        for topic_id, group_df in tqdm(pd_posts.groupby('topic_id'), total=NUM_TOPICS):
            full_text = " . ".join(group_df['cleaned_content'].tolist())
            
            prompt = f"""
Instructions: Read the following social media posts and write a concise, abstractive summary. The summary must be a coherent paragraph that captures the main topic and sentiment of the discussion. Do not just copy and paste sentences.

Posts:
{full_text}

Summary:
"""
            
            try:
                # --- MODIFIED GENERATOR CALL ---
                # We must truncate the *input* (prompt) to the model's 512 token limit.
                # We *only* set max_new_tokens for the *output*.
                summary = generator(
                    prompt,
                    truncation=True,  # Truncate the input prompt
                    max_length=512,  # Max token length for the *input*
                    max_new_tokens=150,  # Max token length for the *output*
                    min_length=30,
                    do_sample=False,
                    no_repeat_ngram_size=2
                )[0]['generated_text']
            
            except Exception as e:
                print(f"Error summarizing topic {topic_id}: {e}")
                summary = f"Error generating summary: {e}"
            
            topic_summaries.append((int(topic_id), summary))
        
        pd_summaries = pd.DataFrame(topic_summaries, columns=["topic_id", "summary_text"])
        pd_mappings = pd_posts[['post_id', 'topic_id']]
        
        print("Analysis complete. Saving results to PostgreSQL...")
        
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                print("Truncating old analysis results...")
                cur.execute("TRUNCATE TABLE global_topics, post_topic_mapping RESTART IDENTITY CASCADE;")
                conn.commit()
        
        pd_summaries.to_sql("global_topics", engine, if_exists='append', index=False)
        pd_mappings.to_sql("post_topic_mapping", engine, if_exists='append', index=False)
        
        print("Successfully saved global topics and post mappings.")
        print("\n--- Final Topic Summaries ---")
        print(pd_summaries.sort_values(by='topic_id').to_string())
        return True
    
    except Exception as e:
        print(f"\n--- 🚨 ERROR during Pandas AI Analysis --- \nError details: {e}")
        return False
    finally:
        if 'engine' in locals():
            engine.dispose()
        print("Analysis process finished.")


def _run_global_analysis_spark(engine):
    """Spark-based distributed topic analysis."""
    print("--- 🚀 Running Spark Analysis ---")
    spark = get_spark_session()
    
    try:
        # Step 1: Load data from PostgreSQL via JDBC
        print("Loading posts from PostgreSQL into Spark...")
        df_posts = spark.read \
            .format("jdbc") \
            .option("url", f"jdbc:postgresql://{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}") \
            .option("dbtable", "posts") \
            .option("user", settings.DB_USER) \
            .option("password", settings.DB_PASS) \
            .load() \
            .filter(F.col("cleaned_content").isNotNull() & (F.col("cleaned_content") != ""))
        
        post_count = df_posts.count()
        if post_count < 20:
            print(f"Not enough posts ({post_count}) to analyze. Aborting.")
            return False
        print(f"Loaded {post_count} posts into Spark DataFrame.")
        
        # Step 2: Generate embeddings in driver (vectorized)
        print("Generating SBERT embeddings...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        posts_list = [row.cleaned_content for row in df_posts.select("cleaned_content").collect()]
        embeddings = sbert_model.encode(posts_list, batch_size=32, show_progress_bar=True)
        np_embeddings = np.array(embeddings)
        
        # Convert embeddings to Spark DataFrame
        print("Converting embeddings to Spark DataFrame...")
        embed_df = spark.createDataFrame(
            [(i, Vectors.dense(vec)) for i, vec in enumerate(np_embeddings)],
            ["index", "features"]
        )
        
        # Step 3: Cluster with Spark MLlib
        print("Preparing embeddings DataFrame: repartitioning and caching...")
        n_partitions = max(2, spark.sparkContext.defaultParallelism)
        embed_df = embed_df.repartition(n_partitions).cache()
        print(f"Partitions: {embed_df.rdd.getNumPartitions()}, Count (materialize): {embed_df.count()}")
        
        # Try Spark MLlib KMeans, but wrap in try/except and fallback to sklearn
        num_topics = settings.SPARK_ANALYSIS_TOPICS
        try:
            print(f"Running Spark MLlib KMeans (k={num_topics}) ...")
            kmeans = SparkKMeans(k=num_topics, seed=42, featuresCol="features",
                                 predictionCol="prediction", maxIter=20)
            model = kmeans.fit(embed_df)
            clustered = model.transform(embed_df).select("index",
                                                         F.col("prediction").alias("topic_id"))
        except Exception as e:
            print("⚠️  Spark KMeans crashed; using sklearn fallback on driver.")
            from sklearn.cluster import KMeans as SkKMeans
            import pandas as pd
            np_emb = np.array([r.features.toArray() for r in embed_df.collect()])
            k_local = min(num_topics, len(np_emb))
            sk = SkKMeans(n_clusters=k_local, random_state=42, n_init=10)
            preds = sk.fit_predict(np_emb)
            clustered = spark.createDataFrame(
                pd.DataFrame({"index": range(len(preds)), "topic_id": preds})
            )
        
        # Step 4: Merge topics back with posts
        posts_with_index = df_posts.withColumn("index", F.monotonically_increasing_id())
        joined_df = posts_with_index.join(clustered, on="index").select("post_id", "cleaned_content", "topic_id")
        
        # Step 5: Summarize each topic using the LLM (in driver)
        generator = get_llm_pipeline()
        print("Generating topic summaries (in driver)...")
        
        summaries = []
        for topic_id in range(num_topics):
            topic_texts = [r.cleaned_content for r in
                           joined_df.filter(F.col("topic_id") == topic_id).select("cleaned_content").collect()]
            if not topic_texts:
                continue
            text_blob = " . ".join(topic_texts)[:4000]
            prompt = f"""
Instructions: Summarize the following posts into a single coherent paragraph describing the main topic.

Posts:
{text_blob}

Summary:
"""
            try:
                summary = \
                generator(prompt, truncation=True, max_length=512, max_new_tokens=150, min_length=30, do_sample=False)[
                    0]['generated_text']
            except Exception as e:
                summary = f"Error generating summary: {e}"
            summaries.append((topic_id, summary))
        
        # Step 6: Save results back to PostgreSQL
        print("Saving global topics and post-topic mappings...")
        pd_summaries = pd.DataFrame(summaries, columns=["topic_id", "summary_text"])
        pd_mappings = joined_df.select("post_id", "topic_id").toPandas()
        
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE global_topics, post_topic_mapping RESTART IDENTITY CASCADE;")
                conn.commit()
        
        pd_summaries.to_sql("global_topics", engine, if_exists='append', index=False)
        pd_mappings.to_sql("post_topic_mapping", engine, if_exists='append', index=False)
        
        print("\n✅ Spark Analysis Complete!")
        return True
    
    except Exception as e:
        print(f"🚨 Spark Analysis Error: {e}")
        return False
    finally:
        if 'spark' in locals():
            print("Spark Analysis finished.")


# (setup_database_tables function is unchanged)
def setup_database_tables():
    print("Setting up database tables for global analysis...")
    
    CREATE_GLOBAL_TOPICS_SQL = """
                               DROP TABLE IF EXISTS post_topic_mapping CASCADE;
                               DROP TABLE IF EXISTS user_topics CASCADE;
                               DROP TABLE IF EXISTS global_topics CASCADE;

                               CREATE TABLE global_topics \
                               ( \
                                   topic_id     INTEGER PRIMARY KEY, \
                                   summary_text TEXT
                               );

                               CREATE TABLE post_topic_mapping \
                               ( \
                                   post_id  TEXT REFERENCES posts (post_id) ON DELETE CASCADE, \
                                   topic_id INTEGER REFERENCES global_topics (topic_id) ON DELETE CASCADE, \
                                   PRIMARY KEY (post_id, topic_id)
                               ); \
                               """
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_GLOBAL_TOPICS_SQL)
                conn.commit()
        print("Successfully dropped old tables and created 'global_topics' and 'post_topic_mapping' tables.")
    except Exception as e:
        print(f"--- 🚨 ERROR: Could not set up database tables --- \nError details: {e}")


if __name__ == "__main__":
    setup_database_tables()
    run_global_analysis()
