# stage_2_content.py

import json
import os
import random
import uuid
from datetime import datetime, timedelta
import networkx as nx
from tqdm import tqdm
import config  # Import our config
from utils import check_file_prerequisites  # Import the checker

# --- Configuration ---
AVG_POSTS_PER_USER = 40

# ... (POST_TEMPLATES and TOPIC_FILLERS dictionaries are unchanged) ...
# (Copy/Paste the full POST_TEMPLATES and TOPIC_FILLERS dicts from your notebook here)
POST_TEMPLATES = {
    "AI Researcher": [
        "Just read a fascinating paper on {topic}. The implications for {field} are huge.",
        "My hot take: {topic} is completely overhyped. The real breakthrough is still 5 years away.",
        "Anyone else attending the {conf} conference? Excited to see the talks on {topic}.",
        "Struggling with this new {library} implementation. Why is {detail} so non-intuitive?",
    ],
    "Data Scientist": [
        "Finished my analysis on {dataset}. Turns out the primary driver for {metric} is {finding}.",
        "Python's {library} is a lifesaver for {task}.",
        "Building a new {model} model to predict {metric}. So far the results are... interesting.",
        "Hot take: 90% of data science is just cleaning {dataset} data.",
    ],
    "Web Developer": [
        "Why did we ever use {old_tech}? {new_tech} is so much cleaner.",
        "Just deployed the new {feature} to prod. Fingers crossed!",
        "TIL about this weird CSS bug in {browser}. Nightmare fuel.",
        "Debating {framework_a} vs {framework_b} for the new project. Thoughts?",
    ],
    "Financial Analyst": [
        "{stock} is looking seriously {sentiment} after their earnings call.",
        "The {market_event} is going to have a major impact on {sector} stocks.",
        "My model predicts a {movement} for {stock} in Q{quarter}.",
        "Deep dive into {company}'s 10-K. Their {metric} looks suspicious.",
    ],
    "NBA Fanatic": [
        "Can you believe that {player} trade? The {team} got fleeced!",
        "{player} is the GOAT, I don't care what anyone says.",
        "That game last night was insane. {team} totally choked in the {quarter}.",
        "My prediction for the finals: {team} vs {team}.",
    ],
    "Indie Gamer": [
        "Just sank 40 hours into {game}. It's a masterpiece of {genre}.",
        "Stop playing {aaa_game} and go play {game}. You won't regret it.",
        "The art style in {game} is just breathtaking.",
        "Shoutout to the solo dev of {game}. Incredible achievement.",
    ],
    "Political Commentator": [
        "The new {policy} is a disaster for {group}.",
        "Can't believe what {politician} said about {issue}. Completely out of touch.",
        "The upcoming {election} is the most important one yet.",
        "Reading the latest poll on {issue}. The numbers are surprising.",
    ],
    "World Traveler": [
        "Just landed in {city}! The {food} is incredible.",
        "Back from {country}. My favorite part was definitely {activity}.",
        "Packing for {country}. Any tips for {activity}?",
        "That {airline} flight was rough, but the view of {landmark} was worth it.",
    ],
    "Aspiring Chef": [
        "Tonight's experiment: {dish} with a {ingredient} twist. It actually worked!",
        "Perfected my {technique} for {food}. The secret is {secret}.",
        "I will never buy {food} from a store again. Homemade is so much better.",
        "Failed attempt at {dish}. It was a {texture} mess.",
    ]
}

TOPIC_FILLERS = {
    "{topic}": ["RAG systems", "scaling laws", "GANs", "customer churn", "React hooks", "CSS grid", "inflation",
                "the playoffs", "Stardew Valley", "immigration policy", "Japan", "sourdough bread"],
    "{field}": ["NLP", "robotics", "e-commerce", "frontend dev", "macroeconomics"],
    "{conf}": ["NeurIPS", "ICLR", "WWDC", "JSConf"],
    "{library}": ["PyTorch", "Pandas", "React", "D3.js"],
    "{detail}": ["the data loader", "the async handling", "the state management"],
    "{dataset}": ["sales data", "user logs", "sensor data"],
    "{metric}": ["conversion rate", "user engagement", "stock price"],
    "{finding}": ["seasonal trends", "user location", "ad spend"],
    "{task}": ["ETL", "data viz", "model training"],
    "{model}": ["regression", "neural net", "prophet"],
    "{old_tech}": ["jQuery", "AngularJS", "legacy PHP"],
    "{new_tech}": ["Svelte", "Next.js", "FastAPI"],
    "{feature}": ["checkout page", "auth flow", "dashboard"],
    "{browser}": ["Safari", "Chrome", "Firefox"],
    "{framework_a}": ["React", "Vue"],
    "{framework_b}": ["Svelte", "Angular"],
    "{stock}": ["TSLA", "AAPL", "GOOGL", "AMZN"],
    "{sentiment}": ["undervalued", "overvalued", "volatile"],
    "{market_event}": ["Fed rate hike", "CPI report"],
    "{sector}": ["tech", "healthcare", "energy"],
    "{movement}": ["10% upside", "20% drop"],
    "{quarter}": ["1", "2", "3", "4"],
    "{company}": ["Enron", "Meta", "startup X"],
    "{player}": ["LeBron", "Jordan", "Curry", "Wemby"],
    "{team}": ["Lakers", "Celtics", "Knicks", "Bulls"],
    "{game}": ["Hades", "Hollow Knight", "Dave the Diver"],
    "{genre}": ["metroidvania", "roguelike", "farming sim"],
    "{aaa_game}": ["Call of Duty", "Assassin's Creed"],
    "{policy}": ["tax bill", "healthcare reform"],
    "{group}": ["small businesses", "students"],
    "{politician}": ["the president", "senator X"],
    "{issue}": ["climate change", "the economy"],
    "{election}": ["midterm", "primary"],
    "{city}": ["Tokyo", "Rome", "Bangkok"],
    "{food}": ["ramen", "pasta", "pad thai"],
    "{country}": ["Italy", "Thailand", "Argentina"],
    "{activity}": ["hiking", "scuba diving", "visiting museums"],
    "{airline}": ["Ryanair", "Spirit"],
    "{landmark}": ["the Alps", "the coastline"],
    "{dish}": ["beef bourguignon", "a souffle", "pho"],
    "{ingredient}": ["cardamom", "truffle oil"],
    "{technique}": ["sous-vide", "maillard reaction"],
    "{secret}": ["more butter", "patience"],
    "{texture}": ["soggy", "dense"]
}


def get_random_timestamp():
    now = datetime.utcnow()
    delta = timedelta(days=random.randint(0, 30),
                      hours=random.randint(0, 23),
                      minutes=random.randint(0, 59))
    return (now - delta).isoformat() + "Z"


def generate_post_content(personas):
    persona = random.choices(personas, weights=[0.7, 0.3], k=1)[0]
    template = random.choice(POST_TEMPLATES[persona])
    content = template
    placeholders = [key for key in TOPIC_FILLERS.keys() if key in content]
    for key in placeholders:
        content = content.replace(key, random.choice(TOPIC_FILLERS[key]), 1)
    return content


def main():
    """Main function to run Stage 2."""
    print("--- Starting Stage 2 ---")
    
    # Prerequisite check
    success, msg = check_file_prerequisites(2)
    if not success:
        print(f"🚨 ERROR: {msg}")
        return
    
    # Load users from Stage 1
    print(f"Loading users from '{config.USERS_JSON_PATH}'...")
    with open(config.USERS_JSON_PATH, 'r') as f:
        users = json.load(f)
    
    # Generate "Original Posts"
    all_posts = []
    print(f"Generating synthetic posts for {len(users)} users...")
    for user in tqdm(users, desc="Generating Posts"):
        user_id = user['user_id']
        personas = user['personas']
        num_posts = random.randint(AVG_POSTS_PER_USER - 15, AVG_POSTS_PER_USER + 15)
        for _ in range(num_posts):
            post_content = generate_post_content(personas)
            all_posts.append({
                "post_id": str(uuid.uuid4()),
                "author_id": user_id,
                "content": post_content,
                "created_at": get_random_timestamp()
            })
    
    # Save Posts to JSON
    print(f"\nSaving {len(all_posts)} posts to '{config.POSTS_JSON_PATH}'...")
    with open(config.POSTS_JSON_PATH, "w") as f:
        json.dump(all_posts, f, indent=2)
    
    # Convert 'follows.edgelist' to 'follows.json'
    print(f"Converting '{config.EDGELIST_PATH}' to '{config.FOLLOWS_JSON_PATH}'...")
    try:
        G = nx.read_edgelist(config.EDGELIST_PATH, create_using=nx.DiGraph())
        follows_list = []
        for follower, followed in G.edges():
            follows_list.append({
                "follower_id": follower,
                "followed_id": followed
            })
        with open(config.FOLLOWS_JSON_PATH, "w") as f:
            json.dump(follows_list, f, indent=2)
        print(f"Successfully converted {len(follows_list)} follow relationships.")
    
    except FileNotFoundError:
        print(f"ERROR: '{config.EDGELIST_PATH}' not found. Did Stage 1 run correctly?")
    
    print("\n--- Stage 2 Complete ---")


if __name__ == "__main__":
    main()
