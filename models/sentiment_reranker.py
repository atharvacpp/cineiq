"""
CineIQ Sentiment-Aware Re-Ranker
=================================
Uses VADER (fast), DistilBERT (accurate), or a Custom Model trained on IMDB 50K
to re-rank recommendations based on real audience reception signals.

Since the IMDB 50K dataset has no movie identifiers, this module:
  1. Can analyse arbitrary review text passed directly
  2. Uses TMDB overview + aggregated metadata as a proxy for movie sentiment
  3. Blends text sentiment score with TMDB vote average.

Re-ranking formula:  final_score = hybrid_score × 0.7 + sentiment_score × 0.3
"""

import os
import pandas as pd
import numpy as np
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'custom_sentiment_model.pkl')

# Lazy-load heavy dependencies
_vader_analyzer = None


def _get_vader():
    """Lazy-load VADER analyzer."""
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader_analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("vaderSentiment not installed. Sentiment scoring will return neutral.")
    return _vader_analyzer


class SentimentReRanker:
    """
    Sentiment-aware re-ranker that adjusts recommendation scores
    based on audience reception analysis.
    """

    def __init__(self, mode='vader'):
        """
        Args:
            mode (str): 'vader', 'distilbert', or 'custom' (IMDB trained model)
        """
        self.movies_df = None
        self.processed_data_path = os.path.join(DATA_DIR, 'processed', 'movies_merged.parquet')
        self.mode = mode
        
        self.sentiment_pipeline = None
        self.custom_model = None

        if self.mode == 'distilbert':
            self._load_distilbert()
        elif self.mode == 'custom':
            self._load_custom_model()

    def _load_distilbert(self):
        """Attempt to load DistilBERT sentiment pipeline."""
        try:
            import torch
            from transformers import pipeline as hf_pipeline

            device = 0 if torch.cuda.is_available() else -1
            device_name = "GPU" if device == 0 else "CPU"
            logger.info(f"Loading DistilBERT model on {device_name}...")
            self.sentiment_pipeline = hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device
            )
            logger.info("DistilBERT loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load DistilBERT: {e}. Falling back to VADER.")
            self.mode = 'vader'

    def _load_custom_model(self):
        """Load the custom sentiment model trained on IMDB 50K."""
        if os.path.exists(CUSTOM_MODEL_PATH):
            try:
                with open(CUSTOM_MODEL_PATH, 'rb') as f:
                    self.custom_model = pickle.load(f)
                logger.info("Custom IMDB sentiment model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load custom model: {e}. Falling back to VADER.")
                self.mode = 'vader'
        else:
            logger.warning(f"Custom model not found at {CUSTOM_MODEL_PATH}. Falling back to VADER. Train it using train_sentiment.py first.")
            self.mode = 'vader'

    def load_data(self):
        """Loads processed metadata."""
        try:
            self.movies_df = pd.read_parquet(self.processed_data_path)
        except Exception as e:
            logger.error(f"Could not load metadata for sentiment analysis: {e}")

    def _get_vader_score(self, text):
        """Returns a sentiment score normalized to [0, 1] using VADER."""
        if not isinstance(text, str) or not text.strip():
            return 0.5

        analyzer = _get_vader()
        if analyzer is None:
            return 0.5

        score = analyzer.polarity_scores(text)['compound']
        return (score + 1) / 2

    def _get_distilbert_score(self, text):
        """Returns an aggregated sentiment score [0, 1] using DistilBERT."""
        if not isinstance(text, str) or not text.strip():
            return 0.5

        if self.sentiment_pipeline is None:
            return self._get_vader_score(text)

        chunk_size = 500
        chunks = [text[i:i + chunk_size] for i in range(0, min(len(text), 2500), chunk_size)]

        scores = []
        for chunk in chunks:
            try:
                result = self.sentiment_pipeline(chunk, truncation=True)[0]
                score = result['score']
                if result['label'] == 'NEGATIVE':
                    score = 1.0 - score
                scores.append(score)
            except Exception as e:
                logger.debug(f"DistilBERT failed on chunk: {e}")

        if not scores:
            return self._get_vader_score(text)

        return sum(scores) / len(scores)

    def _get_custom_score(self, text):
        """Returns a sentiment score [0, 1] using the custom IMDB-trained model."""
        if not isinstance(text, str) or not text.strip():
            return 0.5
            
        if self.custom_model is None:
            return self._get_vader_score(text)
            
        try:
            # Predict probability of positive class
            prob = self.custom_model.predict_proba([text])[0][1]
            return float(prob)
        except Exception as e:
            logger.debug(f"Custom model prediction failed: {e}")
            return self._get_vader_score(text)

    def get_movie_sentiment(self, movie_id):
        """
        Gets the sentiment score for a specific movie.
        Uses the movie's overview text as the basis for sentiment analysis,
        combined with TMDB vote average as a signal.
        """
        if self.movies_df is None:
            self.load_data()

        if self.movies_df is None:
            return 0.5

        movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
        if movie_row.empty:
            return 0.5

        row = movie_row.iloc[0]

        overview = row.get('overview', '')
        if isinstance(overview, str) and overview.strip() and overview.lower() != 'nan':
            if self.mode == 'distilbert':
                text_sentiment = self._get_distilbert_score(overview)
            elif self.mode == 'custom':
                text_sentiment = self._get_custom_score(overview)
            else:
                text_sentiment = self._get_vader_score(overview)
        else:
            text_sentiment = 0.5

        # Blend with TMDB vote average
        tmdb_vote = row.get('tmdb_vote_avg', np.nan)
        if pd.notna(tmdb_vote) and float(tmdb_vote) > 0:
            vote_score = float(tmdb_vote) / 10.0
            return 0.6 * text_sentiment + 0.4 * vote_score
        else:
            return text_sentiment

    def rerank(self, recommendations):
        """
        Takes a list of recommendation dicts and re-ranks them.
        Re-ranking formula: final_score = hybrid_score × 0.7 + sentiment_score × 0.3
        """
        if isinstance(recommendations, list):
            df = pd.DataFrame(recommendations)
        else:
            df = recommendations.copy()

        if df.empty or 'movieId' not in df.columns or 'score' not in df.columns:
            logger.warning("Recommendations are empty or missing required columns.")
            return df

        if self.movies_df is None:
            self.load_data()

        logger.info(f"Computing sentiment scores ({self.mode} mode)...")

        sentiment_scores = []
        for m_id in df['movieId']:
            s_score = self.get_movie_sentiment(m_id)
            sentiment_scores.append(s_score)

        df['sentiment_score'] = sentiment_scores
        df['final_score'] = df['score'] * 0.7 + df['sentiment_score'] * 0.3
        df = df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
        return df

    def analyze_review(self, review_text):
        """Analyzes a single review text and returns sentiment score + label."""
        if self.mode == 'distilbert':
            score = self._get_distilbert_score(review_text)
        elif self.mode == 'custom':
            score = self._get_custom_score(review_text)
        else:
            score = self._get_vader_score(review_text)

        if score >= 0.7:
            label = "positive"
        elif score >= 0.4:
            label = "mixed"
        else:
            label = "negative"

        return {"score": round(score, 4), "label": label, "model": self.mode}


if __name__ == "__main__":
    reranker = SentimentReRanker(mode='vader')
    mock_recs = [
        {"movieId": 1, "title": "Toy Story (1995)", "score": 0.8},
        {"movieId": 2, "title": "Jumanji (1995)", "score": 0.7}
    ]
    df_reranked = reranker.rerank(mock_recs)
    print("Reranked Output:")
    print(df_reranked)
