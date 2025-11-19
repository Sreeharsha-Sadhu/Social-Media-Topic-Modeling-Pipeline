"""
stage1_generate_users.py
------------------------------------------------------------
Stage 1: Synthetic User & Social Graph Generation (Reddit-focused)
Generates:
  - Synthetic users with personas and subreddit interests tuned for topic modelling
  - Directed follow graph reflecting persona similarity
Outputs:
  - users.json
  - follows.edgelist
"""

import json
import os
import random
from typing import List, Dict, Tuple
import networkx as nx

from src.core import config
from src.core.logging_config import get_logger

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Configuration - Reddit-focused personas & subreddits
# -------------------------------------------------------------------
NUM_USERS: int = 25
AVG_FOLLOWS_PER_USER: int = 8
SEED: int = 42  # Set to None to disable deterministic output

PERSONAS: List[str] = [
    "ML Researcher", "Data Journalist", "Front-end Dev",
    "Investor", "NBA Fanatic", "Indie Developer",
    "Political Commentator", "Science Enthusiast", "Movie Buff"
]

# Subreddits chosen for high-quality topical content that supports clustering / summarization
SUBREDDIT_MAP: Dict[str, List[str]] = {
    "ML Researcher": ["MachineLearning", "datascience", "learnmachinelearning", "computervision"],
    "Data Journalist": ["dataisbeautiful", "datascience", "statistics", "python"],
    "Front-end Dev": ["webdev", "reactjs", "frontend", "javascript"],
    "Investor": ["investing", "StockMarket", "economics", "personalfinance"],
    "NBA Fanatic": ["nba", "basketball", "sports"],
    "Indie Developer": ["IndieGaming", "gamedev", "games", "GameDevClassifieds"],
    "Political Commentator": ["politics", "PoliticalDiscussion", "worldnews", "geopolitics"],
    "Science Enthusiast": ["science", "astronomy", "askscience", "Space"],
    "Movie Buff": ["movies", "TrueFilm", "FilmStudies", "criterion"]
}


# -------------------------------------------------------------------
# User Generation
# -------------------------------------------------------------------
def create_users(num_users: int, personas: List[str]) -> List[Dict]:
    """Generate synthetic user profiles (Reddit subscriptions)."""
    if SEED is not None:
        random.seed(SEED)

    logger.info(f"Generating {num_users} synthetic Reddit-style users...")
    users: List[Dict] = []

    for i in range(1, num_users + 1):
        primary = random.choice(personas)
        secondary = random.choice([p for p in personas if p != primary])

        # combine subreddit interests, prefer those from primary persona
        subreddits = list(
            dict.fromkeys(
                SUBREDDIT_MAP[primary] +
                random.sample(SUBREDDIT_MAP[secondary], min(2, len(SUBREDDIT_MAP[secondary])))
            )
        )
        random.shuffle(subreddits)

        users.append({
            "user_id": f"user_{i}",
            "username": f"user_{i}",
            "personas": [primary, secondary],
            # store subreddits as lowercase to match typical reddit names
            "subreddits": [s.lower() for s in subreddits]
        })

    logger.info(f"Generated {len(users)} users.")
    return users


# -------------------------------------------------------------------
# Social Graph Construction
# -------------------------------------------------------------------
def create_social_graph(users: List[Dict], avg_follows: int) -> nx.DiGraph:
    """Build directed persona-weighted follow graph."""
    logger.info("Constructing social graph...")

    G = nx.DiGraph()
    user_ids = [u["user_id"] for u in users]
    G.add_nodes_from(user_ids)

    for user in users:
        user_id = user["user_id"]
        user_personas = set(user["personas"])
        choices = [u for u in users if u["user_id"] != user_id]

        weights = [
            5 if user_personas.intersection(set(other["personas"])) else 1
            for other in choices
        ]

        follow_count = max(1, random.randint(avg_follows - 3, avg_follows + 3))
        selected_targets = random.choices(choices, weights=weights, k=follow_count)

        for target in selected_targets:
            G.add_edge(user_id, target["user_id"])

    logger.info(
        f"Graph built successfully with {G.number_of_nodes()} nodes and "
        f"{G.number_of_edges()} edges."
    )
    return G


# -------------------------------------------------------------------
# Persistence
# -------------------------------------------------------------------
def save_outputs(users: List[Dict], graph: nx.DiGraph) -> None:
    """Save Stage 1 outputs to disk."""
    os.makedirs(config.DATA_DIR, exist_ok=True)

    with open(config.USERS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

    nx.write_edgelist(graph, config.EDGELIST_PATH, data=False)

    logger.info(f"Users saved to {config.USERS_JSON_PATH}")
    logger.info(f"Social graph saved to {config.EDGELIST_PATH}")


# -------------------------------------------------------------------
# Stage 1 Entry Point
# -------------------------------------------------------------------
def main() -> Tuple[List[Dict], nx.DiGraph]:
    logger.info("Starting Stage 1: User & Graph Generation (Reddit-mode)")
    users = create_users(NUM_USERS, PERSONAS)
    graph = create_social_graph(users, AVG_FOLLOWS_PER_USER)
    save_outputs(users, graph)
    logger.info("Stage 1 completed successfully.")
    return users, graph


if __name__ == "__main__":
    main()
