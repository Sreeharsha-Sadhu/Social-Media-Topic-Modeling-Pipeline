"""
stage_1_generate_users.py
----------------------------------
Stage 1: Synthetic User & Social Graph Generation

Generates:
  - A set of synthetic users with assigned personas and subreddit interests.
  - A directed social graph representing user follow relationships.

Outputs:
  - users.json        : List of users and their attributes.
  - follows.edgelist  : Directed graph of follows relationships.
"""

import json
import os
import random
import networkx as nx
from typing import List, Dict, Tuple

from src.core import config
from src.core.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
NUM_USERS: int = 25
AVG_FOLLOWS_PER_USER: int = 8

PERSONAS: List[str] = [
    "AI Researcher", "Data Scientist", "Web Developer",
    "Financial Analyst", "NBA Fanatic", "Indie Gamer",
    "Political Commentator", "World Traveler", "Aspiring Chef"
]

# Subreddit mappings by persona
SUBREDDIT_MAP: Dict[str, List[str]] = {
    "AI Researcher": ["MachineLearning", "LocalLLaMA", "singularity"],
    "Data Scientist": ["datascience", "dataanalysis", "Python"],
    "Web Developer": ["webdev", "reactjs", "node"],
    "Financial Analyst": ["wallstreetbets", "StockMarket", "SecurityAnalysis"],
    "NBA Fanatic": ["nba", "lakers", "bostonceltics"],
    "Indie Gamer": ["IndieGaming", "Games", "StardewValley"],
    "Political Commentator": ["politics", "PoliticalDiscussion", "geopolitics"],
    "World Traveler": ["travel", "solotravel", "digitalnomad"],
    "Aspiring Chef": ["Cooking", "AskCulinary", "Breadit"]
}

# For reproducibility (set to None for random seed)
SEED: int = 42


# ---------------------------------------------------------------------
# User Creation
# ---------------------------------------------------------------------
def create_users(num_users: int, personas: List[str]) -> List[Dict]:
    """
    Generate a list of synthetic users with assigned personas and subreddits.

    Args:
        num_users (int): Number of users to generate.
        personas (List[str]): Available persona categories.

    Returns:
        List[Dict]: List of user objects.
    """
    if SEED is not None:
        random.seed(SEED)

    logger.info(f"Generating {num_users} users with persona diversity...")
    users = []

    for i in range(1, num_users + 1):
        primary = random.choice(personas)
        secondary = random.choice([p for p in personas if p != primary])

        # Combine subreddit interests
        combined_subs = list(set(
            SUBREDDIT_MAP[primary] +
            random.sample(SUBREDDIT_MAP[secondary], 1)
        ))
        random.shuffle(combined_subs)

        users.append({
            "user_id": f"user_{i}",
            "username": f"user_{i}",
            "personas": [primary, secondary],
            "subreddits": combined_subs
        })

    logger.info(f"✅ Generated {len(users)} users successfully.")
    return users


# ---------------------------------------------------------------------
# Graph Creation
# ---------------------------------------------------------------------
def create_social_graph(users: List[Dict], avg_follows: int) -> nx.DiGraph:
    """
    Build a directed graph representing 'follows' relationships between users.

    Args:
        users (List[Dict]): List of user dictionaries.
        avg_follows (int): Average number of follows per user.

    Returns:
        nx.DiGraph: Directed graph object.
    """
    logger.info("Building synthetic social graph...")
    G = nx.DiGraph()
    user_ids = [u["user_id"] for u in users]
    G.add_nodes_from(user_ids)

    for user in users:
        user_id = user["user_id"]
        user_personas = set(user["personas"])
        potential_targets = [u for u in users if u["user_id"] != user_id]

        # Assign higher follow weight for similar personas
        weights = [
            5 if user_personas.intersection(set(pf["personas"])) else 1
            for pf in potential_targets
        ]

        # Variable follow count around average
        num_to_follow = max(1, random.randint(avg_follows - 3, avg_follows + 3))

        if potential_targets:
            selected = random.choices(potential_targets, weights=weights, k=num_to_follow)
            for target in selected:
                G.add_edge(user_id, target["user_id"])

    logger.info(f"✅ Graph built: {G.number_of_nodes()} users, {G.number_of_edges()} edges.")
    return G


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------
def save_outputs(users: List[Dict], graph: nx.DiGraph) -> None:
    """
    Save generated users and follows graph to JSON/edgelist files.

    Args:
        users (List[Dict]): List of user objects.
        graph (nx.DiGraph): Directed follows graph.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)

    users_path = config.USERS_JSON_PATH
    edgelist_path = config.EDGELIST_PATH

    with open(users_path, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

    nx.write_edgelist(graph, edgelist_path, data=False)

    logger.info(f"📁 Users saved → {users_path}")
    logger.info(f"📁 Social graph saved → {edgelist_path}")


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------
def main() -> Tuple[List[Dict], nx.DiGraph]:
    """Execute Stage 1: user generation and social graph construction."""
    logger.info("🚀 Starting Stage 1: Generate Users & Social Graph")
    users = create_users(NUM_USERS, PERSONAS)
    graph = create_social_graph(users, AVG_FOLLOWS_PER_USER)
    save_outputs(users, graph)
    logger.info("🏁 Stage 1 completed successfully.")
    return users, graph


if __name__ == "__main__":
    main()
