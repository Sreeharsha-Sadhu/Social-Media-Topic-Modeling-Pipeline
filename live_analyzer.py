# live_analyzer.py

import pandas as pd
from sklearn.cluster import KMeans as SklearnKMeans
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

import utils
import config
import text_extractor
from stage_4_analysis import get_llm_pipeline  # Reuse our model


def get_user_subreddits(user_id):
    """Fetches the list of subreddits for a user from the DB."""
    try:
        with utils.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT subreddits FROM users WHERE user_id = %s", (user_id,))
                result = cur.fetchone()
                if result:
                    return result[0]
                else:
                    return None
    except Exception as e:
        print(f"🚨 Error fetching user subreddits: {e}")
        return None


def analyze_live_feed(user_id):
    """
    Main function for the new live analysis.
    Fetches, scrapes, clusters, and summarizes live Reddit data.
    """
    print(f"--- 🚀 Starting Live Analysis for {user_id} ---")
    
    subreddits = get_user_subreddits(user_id)
    if not subreddits:
        print(f"Could not find subreddits for {user_id}. Did you run the ETL (Option 4)?")
        return
    
    print(f"Found {len(subreddits)} subreddits: {', '.join(subreddits)}")
    
    try:
        reddit = text_extractor.get_reddit_client()
        posts_to_analyze = []
        
        for sub_name in subreddits:
            print(f"Fetching top 10 posts from {sub_name}...")
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.top(time_filter="week", limit=10):
                content = text_extractor.get_reddit_post_content(post)
                if content:
                    posts_to_analyze.append(content)
    
    except Exception as e:
        print(f"🚨 Error connecting to Reddit API: {e}")
        print("Please check your credentials in config.py.")
        return
    
    if len(posts_to_analyze) < 5:
        print(f"Not enough text content found ({len(posts_to_analyze)} posts) to analyze.")
        return
    
    print(f"\nFound {len(posts_to_analyze)} posts with text. Proceeding with AI analysis...")
    
    print("Loading sentence-transformer model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    print("Generating embeddings...")
    embeddings = sbert_model.encode(
        posts_to_analyze,
        batch_size=16,
        show_progress_bar=True
    )
    
    num_live_topics = max(2, min(5, len(posts_to_analyze) // 5))
    print(f"Clustering posts into {num_live_topics} topics...")
    kmeans = SklearnKMeans(n_clusters=num_live_topics, random_state=0, n_init=10)
    
    pd_live_posts = pd.DataFrame({
        'cleaned_content': posts_to_analyze,
        'topic_id': kmeans.fit_predict(embeddings)
    })
    
    print("Summarizing live topics...")
    generator = get_llm_pipeline()
    
    print("\n--- 📈 Live Analysis Results ---")
    
    for topic_id, group_df in pd_live_posts.groupby('topic_id'):
        
        full_text = " . ".join(group_df['cleaned_content'].tolist())
        truncated_text = full_text[:4000]  # Use full_text for the prompt
        
        prompt = f"""
Instructions: Read the following social media posts and articles, which are all about the same topic. Write a concise, abstractive summary. The summary must be a coherent paragraph that captures the main topic and sentiment of the discussion.

Posts:
{truncated_text}

Summary:
"""
        try:
            # --- MODIFIED GENERATOR CALL ---
            # Same fix as in Stage 4
            summary = generator(
                prompt,
                truncation=True,
                max_length=512,  # Max *input* token limit
                max_new_tokens=150,  # Max *output* token limit
                min_length=30,
                do_sample=False,
                no_repeat_ngram_size=2
            )[0]['generated_text']
        
        except Exception as e:
            print(f"Error summarizing topic {topic_id}: {e}")
            summary = f"Error generating summary: {e}"
        
        print("\n" + "=" * 40)
        print(f"   LIVE TOPIC #{topic_id} (Posts: {len(group_df)})")
        print(f"   SUMMARY: {summary}")
        print("=" * 40)
