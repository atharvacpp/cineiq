"""
CineIQ Explainability Layer
=============================
Every recommendation surfaces a human-readable reason using either:
  1. LIME (Local Interpretable Model-agnostic Explanations) - perturbs text features
     to identify which words drove the similarity score.
  2. Rule-based templates - as a fast fallback using extracted metadata (genres, cast, director).

The Explainer checks the source of each recommendation and generates appropriate explanations.
"""

import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Lazy load LIME
_lime_explainer = None


def _get_lime():
    """Lazy-load LIME text explainer."""
    global _lime_explainer
    if _lime_explainer is None:
        try:
            from lime.lime_text import LimeTextExplainer
            _lime_explainer = LimeTextExplainer(class_names=['Not Similar', 'Similar'])
        except ImportError:
            logger.warning("lime not installed. Using rule-based explanations only.")
    return _lime_explainer


class Explainer:
    """
    Generates human-readable explanations for movie recommendations
    using LIME and rule-based templates.
    """

    def __init__(self, recommender_instance=None):
        """
        Initializes the Explainer.

        Args:
            recommender_instance: An optional HybridRecommender instance
                to reuse the pre-computed TF-IDF matrix and vectorizer.
        """
        self.movies_df = None
        self.recommender = recommender_instance

    def explain(self, recommended_title, query_title=None, user_id=None,
                rec_source='hybrid', sentiment_score=None):
        """
        Generates an explanation string for why a movie was recommended.

        Args:
            recommended_title (str): The movie being recommended.
            query_title (str): The movie the user searched for/liked.
            user_id (str/int): The user's ID.
            rec_source (str): 'content', 'collaborative', or 'hybrid'.
            sentiment_score (float): The sentiment score from the reranker.

        Returns:
            str: A human-readable explanation string.
        """
        if self.recommender is None:
            # Lazy load recommender if not provided
            try:
                from models.recommender import HybridRecommender
                self.recommender = HybridRecommender()
                self.recommender.load_data()
            except Exception as e:
                logger.error(f"Could not load recommender for explanations: {e}")
                return "Recommended based on overall popularity."

        self.movies_df = self.recommender.movies_df
        explanation_parts = []

        # ── 1. Content-Based Explanation ──────────────────────────────────
        if rec_source in ['content', 'hybrid'] and query_title is not None:
            # Try LIME first, fall back to rule-based
            lime_features = self._get_lime_features(query_title, recommended_title)
            if lime_features:
                features_str = ", ".join([f"'{f}'" for f in lime_features])
                explanation_parts.append(
                    f"Because you liked '{query_title}', sharing key elements like {features_str}."
                )
            else:
                # Use rule-based fallback with actual metadata
                shared = self._get_shared_features(query_title, recommended_title)
                if shared:
                    explanation_parts.append(
                        f"Because you liked '{query_title}', sharing {shared}."
                    )
                else:
                    explanation_parts.append(
                        f"Because you liked '{query_title}', which shares similar genres and themes."
                    )

        # ── 2. Collaborative Filtering Explanation ────────────────────────
        if rec_source in ['collaborative', 'hybrid'] and user_id is not None:
            explanation_parts.append("Users with similar taste to you loved this.")

        # ── 3. Sentiment Explanation ──────────────────────────────────────
        if sentiment_score is not None:
            if sentiment_score >= 0.7:
                reception = "highly positive"
            elif sentiment_score >= 0.4:
                reception = "mixed"
            else:
                reception = "negative"
            explanation_parts.append(f"This film has a {reception} audience reception.")

        # ── 4. Fallback ──────────────────────────────────────────────────
        if not explanation_parts:
            return "Recommended based on overall popularity and system metrics."

        return " ".join(explanation_parts)

    def _get_shared_features(self, query_title, recommended_title):
        """
        Rule-based extraction of shared features between two movies.
        Returns a human-readable string of shared genres, cast, or director.
        """
        try:
            df = self.movies_df
            query_row = df[df['title'].str.lower() == str(query_title).lower()]
            rec_row = df[df['title'].str.lower() == str(recommended_title).lower()]

            if query_row.empty or rec_row.empty:
                return None

            query = query_row.iloc[0]
            rec = rec_row.iloc[0]
            shared_items = []

            # Shared genres
            if 'genres' in df.columns:
                q_genres = set(str(query.get('genres', '')).replace('|', ',').split(','))
                r_genres = set(str(rec.get('genres', '')).replace('|', ',').split(','))
                shared_genres = q_genres & r_genres - {'', 'nan', '(no genres listed)'}
                if shared_genres:
                    shared_items.append(f"genres like {', '.join(list(shared_genres)[:2])}")

            # Shared cast members
            if 'cast' in df.columns:
                q_cast = set(str(query.get('cast', '')).split(','))
                r_cast = set(str(rec.get('cast', '')).split(','))
                q_cast = {c.strip() for c in q_cast if c.strip() and c.strip().lower() != 'nan'}
                r_cast = {c.strip() for c in r_cast if c.strip() and c.strip().lower() != 'nan'}
                shared_cast = q_cast & r_cast
                if shared_cast:
                    shared_items.append(f"actor(s) {', '.join(list(shared_cast)[:2])}")

            # Same director
            if 'director' in df.columns:
                q_dir = str(query.get('director', '')).strip()
                r_dir = str(rec.get('director', '')).strip()
                if q_dir and r_dir and q_dir.lower() != 'nan' and q_dir == r_dir:
                    shared_items.append(f"director {q_dir}")

            if shared_items:
                return ', '.join(shared_items)
            return None

        except Exception as e:
            logger.debug(f"Shared feature extraction failed: {e}")
            return None

    def _get_lime_features(self, query_title, recommended_title):
        """
        Uses LIME to extract the top 3 features that drove the similarity score
        between the query movie and recommended movie.
        """
        lime_exp = _get_lime()
        if lime_exp is None:
            return []

        try:
            from sklearn.metrics.pairwise import linear_kernel

            df = self.movies_df
            query_idx = df[df['title'].str.lower() == str(query_title).lower()].index
            rec_idx = df[df['title'].str.lower() == str(recommended_title).lower()].index

            if len(query_idx) == 0 or len(rec_idx) == 0:
                return []

            if self.recommender.tfidf_matrix is None or self.recommender.tfidf_vectorizer is None:
                return []

            query_vec = self.recommender.tfidf_matrix[query_idx[0]]
            rec_soup = df.iloc[rec_idx[0]]['soup']

            # Predictor function for LIME
            def predict_similarity(texts):
                vecs = self.recommender.tfidf_vectorizer.transform(texts)
                sims = linear_kernel(vecs, query_vec).flatten()
                sims = np.clip(sims, 0, 1)
                return np.vstack([1 - sims, sims]).T

            # Generate LIME explanation (small num_samples for API speed)
            exp = lime_exp.explain_instance(
                rec_soup,
                predict_similarity,
                num_features=3,
                num_samples=50
            )

            # Extract the top features driving similarity
            top_features = []
            for feature, weight in exp.as_list():
                if weight > 0:
                    top_features.append(feature)

            return top_features[:3]
        except Exception as e:
            logger.debug(f"LIME explanation failed: {e}")
            return []


if __name__ == "__main__":
    # Test execution
    from models.recommender import HybridRecommender
    rec = HybridRecommender()
    if rec.load_data():
        explainer = Explainer(recommender_instance=rec)
        explanation = explainer.explain(
            "Toy Story 2 (1999)",
            query_title="Toy Story (1995)",
            user_id=1,
            rec_source="hybrid",
            sentiment_score=0.8
        )
        print(f"Explanation: {explanation}")
