# stage_1_generation.py

import networkx as nx
import random
import json
import os
import config  # Import our new config file

# --- Configuration ---
NUM_USERS = 25
PERSONAS = [
    "AI Researcher", "Data Scientist", "Web Developer",
    "Financial Analyst", "NBA Fanatic", "Indie Gamer",
    "Political Commentator", "World Traveler", "Aspiring Chef"
]
AVG_FOLLOWS_PER_USER = 8


def create_users(num_users, personas):
    users = []
    for i in range(1, num_users + 1):
        primary_persona = random.choice(personas)
        secondary_persona = random.choice([p for p in personas if p != primary_persona])
        users.append({
            "user_id": f"user_{i}",
            "username": f"user_{i}",
            "personas": [primary_persona, secondary_persona]
        })
    return users


def create_social_graph(users, avg_follows):
    G = nx.DiGraph()
    user_ids = [u["user_id"] for u in users]
    G.add_nodes_from(user_ids)
    
    for user in users:
        user_id = user["user_id"]
        user_personas = set(user["personas"])
        potential_follows = [u for u in users if u["user_id"] != user_id]
        
        weights = []
        for pf in potential_follows:
            pf_personas = set(pf["personas"])
            if user_personas.intersection(pf_personas):
                weights.append(5)
            else:
                weights.append(1)
        
        num_to_follow = random.randint(avg_follows - 3, avg_follows + 3)
        
        if potential_follows:
            followed_list = random.choices(potential_follows, weights=weights, k=num_to_follow)
            for followed_user in followed_list:
                G.add_edge(user_id, followed_user["user_id"])
    return G


def main():
    """Main function to run Stage 1."""
    print(f"Creating {NUM_USERS} users...")
    users = create_users(NUM_USERS, PERSONAS)
    
    print("Building social graph...")
    social_graph = create_social_graph(users, AVG_FOLLOWS_PER_USER)
    
    print("\n--- Stage 1 Complete ---")
    print(f"Total users created: {len(users)}")
    print(f"Social Graph: {social_graph.number_of_nodes()} nodes, {social_graph.number_of_edges()} edges.")
    
    # Create data directory using config
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Save users list
    with open(config.USERS_JSON_PATH, "w") as f:
        json.dump(users, f, indent=2)
    
    # Save the graph edges
    nx.write_edgelist(social_graph, config.EDGELIST_PATH)
    
    print(f"\nUsers list saved to '{config.USERS_JSON_PATH}'")
    print(f"Social graph saved to '{config.EDGELIST_PATH}'")


if __name__ == "__main__":
    # This allows running the file directly: python stage_1_generation.py
    main()
