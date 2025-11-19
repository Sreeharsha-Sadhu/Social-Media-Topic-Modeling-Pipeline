"""
stage2_generate_posts.py
------------------------------------------------------------
Stage 2: Synthetic Reddit-style Post Generation

Reads:
  - users.json
  - follows.edgelist

Generates:
  - posts.json: each post has fields:
      post_id, author_id, subreddit, title, selftext, content (title + '\n\n' + selftext),
      created_at (ISO), score, num_comments
  - follows.json
"""

import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Any

import networkx as nx
from tqdm import tqdm

from src.core import config
from src.core.logging_config import get_logger
from src.core.utils import check_file_prerequisites

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
AVG_POSTS_PER_USER: int = 40
SEED: int = 42  # deterministic default

# -------------------------------------------------------------------
# Title and body templates tuned to subreddit topics (improves topic modelling)
# -------------------------------------------------------------------
SUBREDDIT_TEMPLATES = {
    "machinelearning": [
        ("New paper on {topic} - thoughts?", "I found this paper discussing {topic} and its implications for {field}. Key points: {finding}."),
        ("How do you handle {detail} in production?", "I'm struggling with {detail} when deploying models. Has anyone used {library} or similar approaches?")
    ],
    "datascience": [
        ("Analysis of {dataset} shows {finding}", "I ran an analysis on {dataset}. The main driver appears to be {finding}. Methods used: {method}."),
        ("Best visualization for {metric}?", "Looking for suggestions to visualize {metric} across categories. Current approach: {approach}.")
    ],
    "python": [
        ("Why does {library} raise {error}?", "I'm getting {error} while using {library}. Minimal example: ..."),
        ("Tips for optimizing {task} in Python", "I've been profiling {task}. Any recommended libraries or patterns?")
    ],
    "technology": [
        ("Big industry update: {event}", "Discussion: {event} and how it affects {sector}. Opinions?"),
        ("Thread: {product} launch analysis", "Short summary of the product features and pros/cons.")
    ],
    "news": [
        ("Breaking: {headline}", "Summary: {headline} — what we know so far: {details}."),
        ("Discussion: {policy} and its impact", "Policy {policy} will likely affect {group}. Thoughts?")
    ],
    "science": [
        ("New study on {subject}", "Paper summary: {finding}. What does this mean for {field}?"),
        ("Question about {concept}", "I'm trying to understand {concept}. Can someone explain?")
    ],
    "politics": [
        ("What do you think about {policy}?", "Short analysis and links: {links}. Repercussions may include {effects}."),
        ("Polling shows {result}", "Interpretation: {interpretation}.")
    ],
    "movies": [
        ("Reaction to {movie} ending", "I just watched {movie} and the ending struck me because {reason}."),
        ("Top films about {topic}", "I recommend these films: {list}.")
    ],
    "investing": [
        ("Is {stock} a buy after {event}?", "Catalysts: {catalyst}. My thesis: {thesis}."),
        ("Market thoughts: {macro_event}", "How might {macro_event} affect asset classes?")
    ],
    "indiegaming": [
        ("Devlog: {game} progress", "Work done this week: {work}. Roadmap: {roadmap}."),
        ("Looking for feedback on gameplay loop", "Short description of the loop and ask for critique.")
    ]
}

FILLERS = {
    "{topic}": ["RAG systems", "scaling laws", "transfer learning", "transformers"],
    "{field}": ["NLP", "computer vision", "time series", "recommendation systems"],
    "{finding}": ["a strong seasonality", "feature leakage", "surprising correlation with location"],
    "{detail}": ["serving latency", "data drift", "memory consumption"],
    "{library}": ["PyTorch", "scikit-learn", "HuggingFace"],
    "{dataset}": ["public sales dataset", "user logs", "open research dataset"],
    "{method}": ["regression", "clustering", "causal inference"],
    "{approach}": ["grouped bar charts", "small multiples"],
    "{error}": ["TypeError", "MemoryError"],
    "{event}": ["major acquisition", "regulatory change", "product recall"],
    "{sector}": ["cloud", "consumer tech", "finance"],
    "{headline}": ["Major diplomatic development in X", "Unexpected election result"],
    "{details}": ["official statements", "witness reports"],
    "{policy}": ["new tax bill", "privacy regulation"],
    "{subject}": ["microbiome diversity", "quantum computing"],
    "{concept}": ["Bayesian priors", "p-hacking"],
    "{product}": ["Widget 2.0", "NewPhone X"],
    "{links}": ["link1, link2", "a thread summarizing the event"],
    "{interpretation}": ["a sign of polarization", "regional shift"],
    "{movie}": ["A Great Film", "A Sci-Fi Hit"],
    "{reason}": ["it reframed the protagonist", "it used time non-linearly"],
    "{list}": ["Film A, Film B, Film C"],
    "{stock}": ["AAPL", "TSLA", "MSFT"],
    "{catalyst}": ["earnings beat", "guidance change"],
    "{thesis}": ["valuation mismatch", "growth runway"],
    "{macro_event}": ["rate hike", "trade tensions"],
    "{game}": ["Tiny Quest", "Rogue Pixel"],
    "{work}": ["AI pathfinding", "art assets"],
    "{roadmap}": ["alpha in 2 months", "demo at conference"]
}


def get_random_timestamp(days_back: int = 14) -> str:
    """Generate a timestamp within the last `days_back` days."""
    now = datetime.now(timezone.utc)
    delta = timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    return (now - delta).isoformat()


def generate_title_and_body(subreddit: str) -> (str, str):
    """Pick a template for the subreddit and fill placeholders."""
    sub = subreddit.lower()
    templates = SUBREDDIT_TEMPLATES.get(sub, SUBREDDIT_TEMPLATES.get("technology"))
    title_tpl, body_tpl = random.choice(templates)
    for placeholder, options in FILLERS.items():
        if placeholder in title_tpl:
            title_tpl = title_tpl.replace(placeholder, random.choice(options), 1)
        if placeholder in body_tpl:
            body_tpl = body_tpl.replace(placeholder, random.choice(options), 1)
    return title_tpl, body_tpl


# -------------------------------------------------------------------
# Main Stage Function
# -------------------------------------------------------------------
def main() -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    logger.info("Starting Stage 2: Generate Reddit-style Posts & Follows")

    if SEED is not None:
        random.seed(SEED)

    ok, msg = check_file_prerequisites(2)
    if not ok:
        logger.error(f"Stage 2 prerequisite failed: {msg}")
        return [], []

    # Load users
    with open(config.USERS_JSON_PATH, "r", encoding="utf-8") as f:
        users = json.load(f)
    logger.info(f"Loaded {len(users)} users for post generation")

    # Generate posts
    all_posts: List[Dict[str, Any]] = []
    logger.info("Generating synthetic Reddit-style posts...")

    for user in tqdm(users, desc="Generating Posts"):
        user_id = user["user_id"]
        subs = user.get("subreddits") or ["technology"]
        num_posts = random.randint(AVG_POSTS_PER_USER - 15, AVG_POSTS_PER_USER + 15)

        for _ in range(num_posts):
            subreddit = random.choice(subs)
            title, selftext = generate_title_and_body(subreddit)
            content = f"{title}\n\n{selftext}"
            post = {
                "post_id": f"reddit_{uuid.uuid4().hex[:12]}",
                "author_id": user_id,
                "subreddit": subreddit,
                "title": title,
                "selftext": selftext,
                "content": content,
                "created_at": get_random_timestamp(days_back=14),
                "score": random.randint(0, 1200),
                "num_comments": random.randint(0, 400)
            }
            all_posts.append(post)

    logger.info(f"Generated {len(all_posts)} total posts.")

    with open(config.POSTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_posts, f, indent=2)
    logger.info(f"Posts saved to {config.POSTS_JSON_PATH}")

    # Convert follows.edgelist to follows.json
    follows_list: List[Dict[str, str]] = []
    try:
        G = nx.read_edgelist(config.EDGELIST_PATH, create_using=nx.DiGraph())
        for follower, followed in G.edges():
            follows_list.append({
                "follower_id": follower,
                "followed_id": followed
            })

        with open(config.FOLLOWS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(follows_list, f, indent=2)

        logger.info(
            f"Follows saved to {config.FOLLOWS_JSON_PATH} "
            f"({len(follows_list)} relationships)"
        )

    except FileNotFoundError:
        logger.error(f"Missing {config.EDGELIST_PATH}. Did Stage 1 run correctly?")

    logger.info("Stage 2 completed successfully.")
    return all_posts, follows_list


if __name__ == "__main__":
    main()
