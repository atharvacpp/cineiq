"""
CineIQ Hybrid Recommendation Engine
====================================
Combines Content-Based Filtering (TF-IDF + Cosine Similarity) with
Collaborative Filtering (Surprise SVD) via a weighted ensemble.

Architecture:
  - Content-Based:  TF-IDF on a "soup" of genres + cast + keywords + overview
  - Collaborative:  SVD from the Surprise library on MovieLens 25M ratings
  - Hybrid:         40% Content + 60% CF (normalized scores)
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import defaultdict
import pickle
import logging

try:
    from surprise import Dataset, Reader, SVD, accuracy
    from surprise.model_selection import train_test_split
    SURPRISE_INSTALLED = True
except ImportError:
    SURPRISE_INSTALLED = False

try:
    import mlflow
    MLFLOW_INSTALLED = True
except ImportError:
    MLFLOW_INSTALLED = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')


class HybridRecommender:
    """
    Hybrid movie recommendation engine that blends content-based and
    collaborative filtering approaches into a single unified scorer.
    """

    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.svd_model = None
        self.tfidf_vectorizer = None

        self.processed_data_path = os.path.join(DATA_DIR, 'processed', 'movies_merged.parquet')
        self.ratings_path = os.path.join(DATA_DIR, 'raw', 'ratings.csv')
        self.model_path = os.path.join(BASE_DIR, 'models', 'svd_model.pkl')

    def load_data(self):
        """Loads processed dataset and prepares Content-Based metadata (TF-IDF)."""
        logger.info("Loading metadata for Content-Based Filter...")
        try:
            self.movies_df = pd.read_parquet(self.processed_data_path)
            self.movies_df = self.movies_df.reset_index(drop=True)
        except Exception as e:
            logger.error(f"Could not load processed metadata: {e}")
            return False

        logger.info("Creating content soup...")

        def create_soup(row):
            """Concatenate all text features into a single 'soup' string for TF-IDF."""
            soup_elements = []
            for col in ['genres', 'cast', 'keywords', 'overview', 'director']:
                if col in row and pd.notnull(row[col]):
                    val = str(row[col])
                    # Replace separators with spaces for uniform tokenization
                    val = val.replace('|', ' ').replace(',', ' ')
                    soup_elements.append(val)
            return ' '.join(soup_elements)

        self.movies_df['soup'] = self.movies_df.apply(create_soup, axis=1)

        logger.info("Fitting TF-IDF Vectorizer...")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['soup'])
        logger.info(f"Content-Based setup complete. TF-IDF shape: {self.tfidf_matrix.shape}")
        return True

    def train_collaborative(self, n_factors=100, n_epochs=20, lr_all=0.005,
                            sample_size=None):
        """
        Trains Collaborative Filter (SVD) and logs to MLflow.

        Args:
            n_factors: Number of latent factors for SVD.
            n_epochs:  Number of training epochs.
            lr_all:    Learning rate for all parameters.
            sample_size: If set, sample this many ratings for faster training.
        """
        if not SURPRISE_INSTALLED:
            logger.error("scikit-surprise is not installed. Skipping SVD training.")
            return False

        logger.info("Loading ratings for Collaborative Filter...")
        try:
            ratings_df = pd.read_csv(self.ratings_path)
        except Exception as e:
            logger.error(f"Could not load ratings: {e}")
            return False

        # Optionally sample for faster training
        if sample_size and len(ratings_df) > sample_size:
            logger.info(f"Sampling {sample_size} ratings from {len(ratings_df)} total...")
            ratings_df = ratings_df.sample(n=sample_size, random_state=42)

        logger.info(f"Preparing data for Surprise SVD ({len(ratings_df)} ratings)...")
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

        # ── MLflow experiment tracking ────────────────────────────────────
        use_mlflow = MLFLOW_INSTALLED
        if use_mlflow:
            try:
                mlruns_dir = os.path.join(BASE_DIR, 'mlruns')
                mlflow.set_tracking_uri(f"file://{mlruns_dir}")
                mlflow.set_experiment("CineIQ_Collaborative_Filtering")
            except Exception as e:
                logger.warning(f"MLflow setup failed: {e}. Training will proceed without tracking.")
                use_mlflow = False

        run_context = mlflow.start_run() if use_mlflow else None

        try:
            if use_mlflow:
                run_context.__enter__()
                mlflow.log_params({
                    "n_factors": n_factors,
                    "n_epochs": n_epochs,
                    "lr_all": lr_all,
                    "num_ratings": len(ratings_df),
                })

            logger.info("Training SVD model...")
            self.svd_model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all,
                                random_state=42)
            self.svd_model.fit(trainset)

            logger.info("Evaluating SVD model...")
            predictions = self.svd_model.test(testset)
            rmse = accuracy.rmse(predictions, verbose=False)

            # Compute Precision@5 and Recall@5
            precisions, recalls = self._precision_recall_at_k(predictions, k=5, threshold=3.5)
            avg_precision = sum(prec for prec in precisions.values()) / max(len(precisions), 1)
            avg_recall = sum(rec for rec in recalls.values()) / max(len(recalls), 1)

            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  Precision@5: {avg_precision:.4f}")
            logger.info(f"  Recall@5: {avg_recall:.4f}")

            if use_mlflow:
                mlflow.log_metrics({
                    "rmse": rmse,
                    "precision_at_5": avg_precision,
                    "recall_at_5": avg_recall
                })

            # Save the trained model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.svd_model, f)

            if use_mlflow:
                mlflow.log_artifact(self.model_path)

            logger.info("Collaborative model trained and saved successfully.")

        finally:
            if use_mlflow and run_context:
                run_context.__exit__(None, None, None)

        return True

    def _precision_recall_at_k(self, predictions, k=10, threshold=3.5):
        """Helper to calculate Precision/Recall @ K."""
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            n_rel_and_rec_k = sum(
                ((true_r >= threshold) and (est >= threshold))
                for (est, true_r) in user_ratings[:k]
            )

            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        return precisions, recalls

    def get_content_similar(self, movie_title, top_n=10):
        """Returns Content-Based recommendations using TF-IDF cosine similarity."""
        if self.movies_df is None or self.tfidf_matrix is None:
            self.load_data()

        # Try exact match first
        idx_list = self.movies_df[
            self.movies_df['title'].str.lower() == str(movie_title).lower()
        ].index
        if len(idx_list) == 0:
            # Fallback to substring match
            matched = self.movies_df[
                self.movies_df['title'].str.contains(
                    str(movie_title), case=False, regex=False, na=False
                )
            ]
            if matched.empty:
                logger.warning(f"Movie '{movie_title}' not found in database.")
                return []
            idx = matched.index[0]
            logger.info(f"Resolved '{movie_title}' to '{self.movies_df.iloc[idx]['title']}'")
        else:
            idx = idx_list[0]

        # Compute cosine similarity
        cosine_sim = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()

        # Sort and get top N (ignore the movie itself)
        movie_indices = cosine_sim.argsort()[:-(top_n + 2):-1]
        movie_indices = [i for i in movie_indices if i != idx][:top_n]

        results = []
        for i in movie_indices:
            results.append({
                "movieId": int(self.movies_df.iloc[i]['movieId']),
                "title": self.movies_df.iloc[i]['title'],
                "score": float(cosine_sim[i])
            })
        return results

    def get_cf_recommendations(self, user_id, top_n=10):
        """Returns Collaborative Filtering recommendations using trained SVD model."""
        if not SURPRISE_INSTALLED:
            return []

        if self.svd_model is None:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.svd_model = pickle.load(f)
            else:
                logger.warning("SVD model not found. Please train it first.")
                return []

        if self.movies_df is None:
            self.load_data()

        all_movie_ids = self.movies_df['movieId'].unique()
        predictions = []
        for m_id in all_movie_ids:
            pred = self.svd_model.predict(user_id, m_id)
            predictions.append((m_id, pred.est))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:top_n]

        results = []
        for m_id, est in top_predictions:
            title = self.movies_df[self.movies_df['movieId'] == m_id]['title'].values
            title = title[0] if len(title) > 0 else "Unknown"
            results.append({
                "movieId": int(m_id),
                "title": title,
                "score": float(est)
            })
        return results

    def get_hybrid_recommendations(self, user_id, movie_title, top_n=10):
        """
        Returns Hybrid Ensemble recommendations (40% Content + 60% CF).

        Falls back to pure content-based if collaborative filtering is unavailable.
        """
        if self.movies_df is None or self.tfidf_matrix is None:
            self.load_data()

        if not SURPRISE_INSTALLED:
            logger.warning("CF unavailable due to missing scikit-surprise. Falling back to Pure Content-Based.")
            content_recs = self.get_content_similar(movie_title, top_n)
            for r in content_recs:
                r['source_weights'] = {"content_score": r['score'], "cf_score": 0.0}
            return content_recs

        if self.svd_model is None:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.svd_model = pickle.load(f)
            else:
                logger.warning("SVD model not found. Falling back to pure content-based.")
                content_recs = self.get_content_similar(movie_title, top_n)
                for r in content_recs:
                    r['source_weights'] = {"content_score": r['score'], "cf_score": 0.0}
                return content_recs

        # Get Content Scores
        idx_list = self.movies_df[
            self.movies_df['title'].str.lower() == str(movie_title).lower()
        ].index
        if len(idx_list) == 0:
            matched = self.movies_df[
                self.movies_df['title'].str.contains(
                    str(movie_title), case=False, regex=False, na=False
                )
            ]
            if matched.empty:
                logger.warning(f"Movie '{movie_title}' not found. Falling back to CF only.")
                return self.get_cf_recommendations(user_id, top_n)
            idx = matched.index[0]
        else:
            idx = idx_list[0]

        cosine_sim = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()

        # Normalize Content Scores to [0, 1]
        content_max = cosine_sim.max()
        content_norm = cosine_sim / content_max if content_max > 0 else cosine_sim

        # Get CF Scores
        all_movie_ids = self.movies_df['movieId'].values
        cf_scores = np.zeros(len(all_movie_ids))

        for i, m_id in enumerate(all_movie_ids):
            cf_scores[i] = self.svd_model.predict(user_id, m_id).est

        # Normalize CF scores [0.5, 5.0] to [0, 1]
        cf_norm = (cf_scores - 0.5) / 4.5

        # Blend (40% Content, 60% CF)
        hybrid_scores = 0.4 * content_norm + 0.6 * cf_norm

        # Sort and exclude the target movie
        movie_indices = hybrid_scores.argsort()[:-(top_n + 2):-1]
        movie_indices = [i for i in movie_indices if i != idx][:top_n]

        results = []
        for i in movie_indices:
            results.append({
                "movieId": int(self.movies_df.iloc[i]['movieId']),
                "title": self.movies_df.iloc[i]['title'],
                "score": float(hybrid_scores[i]),
                "source_weights": {
                    "content_score": float(content_norm[i]),
                    "cf_score": float(cf_norm[i])
                }
            })
        return results

    def get_user_taste_profile(self, user_id):
        """
        Generates a user taste profile from their high-rated (≥4.0) movies.
        Returns top genres, decades, directors, and actors.
        """
        if self.movies_df is None:
            self.load_data()

        if not os.path.exists(self.ratings_path):
            return None

        ratings_df = pd.read_csv(self.ratings_path)
        user_ratings = ratings_df[ratings_df['userId'] == user_id]

        if user_ratings.empty:
            return None

        # Merge with movies to get metadata
        user_movies = user_ratings.merge(self.movies_df, on='movieId', how='left')
        user_movies = user_movies[user_movies['rating'] >= 4.0]

        if user_movies.empty:
            return None

        # 1. Top Genres
        genres_list = []
        for g_str in user_movies['genres'].dropna():
            genres_list.extend([g.strip() for g in str(g_str).replace('|', ',').split(',')])
        # Remove empty strings and "(no genres listed)"
        genres_list = [g for g in genres_list if g and g != '(no genres listed)']
        genre_counts = pd.Series(genres_list).value_counts()
        top_genres = genre_counts.head(5).index.tolist()

        # 2. Top Decades (extract year from title e.g., "Toy Story (1995)")
        import re
        years = user_movies['title'].str.extract(r'\((\d{4})\)')
        if not years.empty and 0 in years.columns:
            years_clean = pd.to_numeric(years[0], errors='coerce').dropna()
            decades = (years_clean // 10) * 10
            top_decades = [f"{int(d)}s" for d in decades.value_counts().head(3).index.tolist()]
        else:
            top_decades = []

        # 3. Top Actors
        top_actors = []
        if 'cast' in user_movies.columns:
            actors_list = []
            for cast_str in user_movies['cast'].dropna():
                actors_list.extend([a.strip() for a in str(cast_str).split(',')])
            actors_list = [a for a in actors_list if a and a.lower() != 'nan']
            if actors_list:
                top_actors = pd.Series(actors_list).value_counts().head(5).index.tolist()

        # 4. Top Directors
        top_directors = []
        if 'director' in user_movies.columns:
            dir_list = []
            for d_str in user_movies['director'].dropna():
                d = str(d_str).strip()
                if d and d.lower() != 'nan':
                    dir_list.append(d)
            if dir_list:
                top_directors = pd.Series(dir_list).value_counts().head(3).index.tolist()

        return {
            "userId": user_id,
            "top_genres": top_genres,
            "top_decades": top_decades,
            "top_directors": top_directors if top_directors else ["No director data available"],
            "top_actors": top_actors if top_actors else ["No cast data available"],
            "total_rated": len(user_ratings),
            "highly_rated": len(user_movies),
            "genre_distribution": genre_counts.head(10).to_dict() if not genre_counts.empty else {},
        }


if __name__ == "__main__":
    recommender = HybridRecommender()
    if recommender.load_data():
        # Quick test: content-based similarity
        results = recommender.get_content_similar("Toy Story (1995)", top_n=5)
        print("\nContent-Based Recommendations for 'Toy Story (1995)':")
        for r in results:
            print(f"  {r['title']} (score: {r['score']:.4f})")
