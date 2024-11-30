import networkx as nx
import time
import sys
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

def build_graph(ratings):
    G = nx.Graph()

    print("Adding users and movies as nodes...")
    time.sleep(1)  # Simulate loading time
    # Add users and movies as nodes
    G.add_nodes_from(ratings['userId'].unique(), bipartite=0)  # Users
    G.add_nodes_from(ratings['movieId'].unique(), bipartite=1)  # Movies
    print("Nodes added successfully!")

    print("Adding edges with weights as ratings...")
    time.sleep(1)  # Simulate loading time
    # Add edges with weights as ratings
    for i, (_, row) in enumerate(ratings.iterrows(), 1):
        G.add_edge(row['userId'], row['movieId'], weight=row['rating'])
        if i % 100 == 0:
            sys.stdout.write(f"\rProcessed {i} edges...")
            sys.stdout.flush()
    print("\nEdges added successfully!")
    return G

def recommend_movies(graph, user_id, top_n=5):
    print(f"Calculating Personalized PageRank for user {user_id}...")
    time.sleep(1)  # Simulate loading time
    # Calculate Personalized PageRank
    personalized_pagerank = nx.pagerank(graph, personalization={user_id: 1})
    print("Personalized PageRank calculated.")

    print("Filtering and sorting movie recommendations...")
    time.sleep(1)  # Simulate loading time
    # Filter out movies that the user has already rated
    user_rated_movies = set(graph.neighbors(user_id))
    movie_scores = {movie: score for movie, score in personalized_pagerank.items() if movie not in user_rated_movies and graph.nodes[movie].get('bipartite') == 1}
    
    # Sort movies by score
    sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
    print("Sorting completed.")

    print(f"Top {top_n} recommendations for user {user_id}:")
    top_recommendations = [int(movie) for movie, score in sorted_movies[:top_n]]  # Convert to native int
    for i, movie in enumerate(top_recommendations, 1):
        print(f"{i}. Movie ID: {movie}")
    return top_recommendations
