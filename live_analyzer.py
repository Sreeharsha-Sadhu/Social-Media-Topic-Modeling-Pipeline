# live_analyzer.py

import pandas as pd
from sklearn.cluster import KMeans as SklearnKMeans
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import asyncio

import utils
import config
import text_extractor
from stage_4_analysis import get_llm_pipeline  # Reuse our LLM
import live_linkedin  # <-- ADDED
import live_twitter  # <-- ADDED


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


def _run_ai_on_text_list(posts_to_analyze: list, num_topics=5):
    """
    Private helper function to run the common AI pipeline
    on a simple list of text strings.
    """
    if len(posts_to_analyze) < 5:
        print(f"Not enough text content found ({len(posts_to_analyze)} posts) to analyze.")
        return
    
    print(f"\nFound {len(posts_to_analyze)} posts with text. Proceeding with AI analysis...")
    
    # 3a. Embedding
    print("Loading sentence-transformer model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    print("Generating embeddings...")
    embeddings = sbert_model.encode(
        posts_to_analyze,
        batch_size=16,
        show_progress_bar=True
    )
    
    # 3b. Clustering
    num_clusters = max(2, min(num_topics, len(posts_to_analyze) // 5))
    print(f"Clustering posts into {num_clusters} topics...")
    kmeans = SklearnKMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    
    pd_live_posts = pd.DataFrame({
        'cleaned_content': posts_to_analyze,
        'topic_id': kmeans.fit_predict(embeddings)
    })
    
    # 3c. Summarization
    print("Summarizing live topics...")
    generator = get_llm_pipeline()  # Get the cached LLM
    
    print("\n--- 📈 Live Analysis Results ---")
    
    for topic_id, group_df in pd_live_posts.groupby('topic_id'):
        
        full_text = " . ".join(group_df['cleaned_content'].tolist())
        truncated_text = full_text[:4000]
        
        prompt = f"""
Instructions: Read the following social media posts and articles, which are all about the same topic. Write a concise, abstractive summary. The summary must be a coherent paragraph that captures the main topic and sentiment of the discussion.

Posts:
{truncated_text}

Summary:
"""
        try:
            summary = generator(
                prompt,
                truncation=True,
                max_length=512,
                max_new_tokens=150,
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


def analyze_reddit_feed(user_id):
    """
    Analyzes a user's live Reddit feed.
    """
    print(f"--- 🚀 Starting Live Reddit Analysis for {user_id} ---")
    
    subreddits = get_user_subreddits(user_id)
    if not subreddits:
        print(f"Could not find subreddits for {user_id}. Did you run the ETL (Option 4)?")
        return
    
    print(f"Found {len(subreddits)} subreddits: {', '.join(subreddits)}")
    
    try:
        reddit = text_extractor.get_reddit_client()
        posts_to_analyze = []
        
        for sub_name in subreddits:
            print(f"Fetching top 10 posts from r/{sub_name}...")
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.top(time_filter="week", limit=10):
                content = text_extractor.get_reddit_post_content(post)
                if content:
                    posts_to_analyze.append(content)
    
    except Exception as e:
        print(f"🚨 Error connecting to Reddit API: {e}")
        print("Please check your credentials in config.py.")
        return
    
    # Run the common AI pipeline
    _run_ai_on_text_list(posts_to_analyze)


# --- NEW FUNCTION ---
def analyze_linkedin_feed(username, password):
    """
    Analyzes a user's live LinkedIn feed.
    """
    print(f"--- 🚀 Starting Live LinkedIn Analysis ---")
    
    # 1. Scrape data
    # We must run the async function and wait for it to complete
    posts_data = asyncio.run(live_linkedin.scrape_linkedin_feed(username, password))
    
    if not posts_data:
        print("No posts were scraped from LinkedIn. Aborting analysis.")
        return
    
    # 2. Extract text for the AI
    posts_to_analyze = [post['content'] for post in posts_data]
    
    # 3. Run the common AI pipeline
    _run_ai_on_text_list(posts_to_analyze)


# --- NEW FUNCTION ---
def analyze_twitter_feed(username, password):
    """
    Analyzes a user's live Twitter/X feed.
    """
    print(f"--- 🚀 Starting Live Twitter/X Analysis ---")
    
    # 1. Scrape data
    posts_data = asyncio.run(live_twitter.scrape_twitter_feed(username, password))
    
    if not posts_data:
        print("No posts were scraped from Twitter. Aborting analysis.")
        return
    
    # 2. Extract text for the AI
    posts_to_analyze = [post['content'] for post in posts_data]
    
    # 3. Run the common AI pipeline
    _run_ai_on_text_list(posts_to_analyze)
