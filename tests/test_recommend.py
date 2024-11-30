import unittest
import pandas as pd
from src.data_processing import preprocess_data
from src.collaborative import user_based_recommendations
from src.graph_analysis import build_graph, recommend_movies

class TestRecommendationSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Load sample data for testing.
        """
        cls.movies_data = pd.DataFrame({
            "movieId": [1, 2, 3, 4, 5],
            "title": [
                "Toy Story (1995)",
                "Jumanji (1995)",
                "Grumpier Old Men (1995)",
                "Waiting to Exhale (1995)",
                "Father of the Bride Part II (1995)"
            ]
        })
        cls.ratings_data = pd.DataFrame({
            "userId": [1, 1, 1, 2, 2, 3],
            "movieId": [1, 2, 3, 1, 4, 5],
            "rating": [4.0, 5.0, 3.0, 4.0, 5.0, 2.0]
        })

    def test_preprocess_data(self):
        """
        Test preprocessing of data.
        """
        user_movie_matrix = preprocess_data(self.ratings_data, self.movies_data)
        self.assertEqual(user_movie_matrix.shape, (3, 5))  # 3 users, 5 movies
        self.assertEqual(user_movie_matrix.loc[1, 1], 4.0)  # User 1 rated Movie 1 with 4.0

    def test_user_based_recommendations(self):
        """
        Test user-based collaborative filtering recommendations.
        """
        user_movie_matrix = preprocess_data(self.ratings_data, self.movies_data)
        recommendations = user_based_recommendations(user_id=2, user_movie_matrix=user_movie_matrix)
        # Assert recommendations are generated as a list of scores
        self.assertEqual(len(recommendations), 5)
        self.assertGreaterEqual(recommendations[1], 0)  # Score for a movie must be non-negative

    def test_graph_construction(self):
        """
        Test graph construction from ratings data.
        """
        graph = build_graph(self.ratings_data)
        self.assertEqual(len(graph.nodes), 8)  # 3 users + 5 movies

    def test_graph_based_recommendations(self):
        """
        Test graph-based recommendations.
        """
        graph = build_graph(self.ratings_data)
        recommendations = recommend_movies(graph, user_id=2)
        recommended_titles = self.movies_data[self.movies_data['movieId'].isin(recommendations)]['title'].tolist()
        self.assertIn("Waiting to Exhale (1995)", recommended_titles)

if __name__ == "__main__":
    unittest.main()
