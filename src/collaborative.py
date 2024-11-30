from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

def user_based_recommendations(user_id: int, user_item_matrix, k=5):
    print(f"Starting recommendation process for user_id: {user_id}")
    
    user_id = int(user_id)  # Ensure user_id is an integer
    print("Converted user_id to integer.")
    
    user_item_matrix = user_item_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)  # Ensure only numerical values
    print("Converted user_item_matrix to numerical values and filled NaNs with 0.")
    
    if user_id < 1 or user_id > user_item_matrix.shape[0] - 1:
        raise ValueError("user_id is out of bounds")
    if user_item_matrix.size == 0:
        raise ValueError("user_item_matrix is empty")
    
    user_item_matrix = csr_matrix(user_item_matrix)  # Convert to sparse matrix
    print("Converted user_item_matrix to sparse matrix.")
    
    print("Calculating cosine similarity between users...")
    user_idx = user_id - 1  # Adjust index to match matrix
    user_similarity = cosine_similarity(user_item_matrix[user_idx], user_item_matrix).flatten()
    print("Cosine similarity calculation complete.")
    
    similar_users = np.argsort(-user_similarity)[1:k+1]  # Top-k neighbors
    similar_users = [idx for idx in similar_users if idx < user_item_matrix.shape[0] and idx >= 0]  # Ensure indices are within bounds
    print(f"Identified top {k} similar users: {similar_users}")
    
    print("Aggregating ratings from similar users...")
    recommendations = user_item_matrix[similar_users].mean(axis=0).A1
    recommendations[user_idx] = 0  # Exclude the user's own ratings
    print("Aggregation complete.")
    
    return np.asarray(recommendations).flatten()

def format_recommendations(recommendations, movies_df):
    """
    Map recommendation scores to movie titles.
    """
    print("Formatting recommendations...")
    movie_recommendations = []
    for movie_id, score in enumerate(recommendations):
        if score > 0:  # Only include movies with a positive score
            movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
            movie_recommendations.append((movie_title, score))
            print(f"Added recommendation: {movie_title} with score {score}")
    
    sorted_recommendations = sorted(movie_recommendations, key=lambda x: -x[1])[:10]  # Top 10
    print("Recommendations formatted and sorted.")
    return sorted_recommendations
