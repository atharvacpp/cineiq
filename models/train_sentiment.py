"""
CineIQ Sentiment Model Training
===============================
Trains a custom sentiment classification model using the IMDB 50K Reviews dataset.
This fulfills the requirement to use the IMDB dataset for sentiment model training.

The trained model is a lightweight TF-IDF + Logistic Regression pipeline,
which is extremely fast for inference during recommendations.
"""

import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'custom_sentiment_model.pkl')

def train_model():
    """Trains a sentiment classifier on IMDB 50K reviews."""
    imdb_path = os.path.join(DATA_DIR, 'raw', 'imdb_reviews.csv')
    
    if not os.path.exists(imdb_path):
        logger.error(f"IMDB reviews dataset not found at {imdb_path}.")
        logger.error("Please download it from Kaggle and place it in data/raw/")
        return False
        
    logger.info("Loading IMDB 50K Reviews dataset...")
    df = pd.read_csv(imdb_path)
    
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        logger.error("Dataset missing 'review' or 'sentiment' columns.")
        return False
        
    logger.info(f"Dataset loaded: {len(df)} reviews.")
    
    # Map sentiment to binary labels
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df = df.dropna(subset=['label'])
    
    # Optional: Subsample for faster training during dev/testing
    # df = df.sample(n=10000, random_state=42)
    
    X = df['review']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info("Training TF-IDF + Logistic Regression pipeline...")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=500, n_jobs=-1, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    logger.info("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
        
    logger.info(f"Model saved successfully to {MODEL_PATH}")
    return True

if __name__ == "__main__":
    train_model()
