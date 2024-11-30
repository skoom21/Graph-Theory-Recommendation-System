import pandas as pd

def load_data(ratings_path, movies_path):
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    return ratings, movies

def preprocess_data(ratings, movies):
    # Merge datasets if needed
    data = ratings.merge(movies, on='movieId')
    return data
