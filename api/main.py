"""
CineIQ FastAPI Backend
======================
Serves the hybrid recommendation engine, content similarity,
user taste profiles, and sentiment analysis via REST API.

Endpoints:
  GET /recommend    - Hybrid recommendations with explanations
  GET /similar      - Pure content-based similarities
  GET /user-profile - User taste profile (genres, decades, directors, actors)
  POST /analyze-sentiment - Analyze sentiment of a review text
"""

import os
import sys
import pandas as pd
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.recommender import HybridRecommender
from models.sentiment_reranker import SentimentReRanker
from models.explainer import Explainer

# ── Global model instances ────────────────────────────────────────────────
recommender = HybridRecommender()
reranker = SentimentReRanker(mode='vader')
explainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads datasets and models into memory on startup."""
    global explainer
    print("=" * 50)
    print("Loading CineIQ datasets and models...")
    recommender.load_data()
    reranker.load_data()
    explainer = Explainer(recommender_instance=recommender)
    print("Models loaded successfully.")
    print("=" * 50)
    yield
    print("CineIQ API shutting down.")


app = FastAPI(
    title="CineIQ API",
    description="Explainable Hybrid Movie Recommender API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Schemas ──────────────────────────────────────────────────────
class RecommendationItem(BaseModel):
    movieId: int
    title: str
    score: float
    explanation: str
    poster_url: str


class UserProfile(BaseModel):
    userId: int
    top_genres: List[str]
    top_decades: List[str]
    top_directors: List[str]
    top_actors: List[str]
    total_rated: Optional[int] = None
    highly_rated: Optional[int] = None
    genre_distribution: Optional[Dict[str, int]] = None


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    score: float
    label: str


class HealthResponse(BaseModel):
    status: str
    movies_loaded: int
    svd_model_available: bool


# ── Helper Functions ──────────────────────────────────────────────────────
def get_poster_url(movie_id: int) -> str:
    """Returns a poster URL placeholder. In production, this would fetch from TMDB API."""
    return f"https://via.placeholder.com/300x450.png?text=Movie+{movie_id}"


# ── API Endpoints ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="ok",
        movies_loaded=len(recommender.movies_df) if recommender.movies_df is not None else 0,
        svd_model_available=recommender.svd_model is not None,
    )


@app.get("/recommend", response_model=List[RecommendationItem])
def get_recommendations(
    user_id: int = Query(..., description="User ID"),
    movie_title: str = Query(..., description="Target movie title the user liked"),
    top_n: int = Query(10, description="Number of recommendations", ge=1, le=50),
):
    """Returns hybrid recommendations with explanations and sentiment re-ranking."""
    try:
        # 1. Get Hybrid Recommendations (fetch extra to allow reranking to shift items)
        raw_recs = recommender.get_hybrid_recommendations(user_id, movie_title, top_n=top_n * 2)
        if not raw_recs:
            raise HTTPException(
                status_code=404,
                detail=f"Movie '{movie_title}' not found or model uninitialized. Try a different title."
            )

        # 2. Sentiment Re-Rank
        reranked_df = reranker.rerank(raw_recs)

        # 3. Generate Explanations and Format
        final_recs = []
        for _, row in reranked_df.head(top_n).iterrows():
            m_id = int(row['movieId'])
            t = row['title']
            s_score = row.get('sentiment_score', 0.5)
            f_score = row.get('final_score', row['score'])

            explanation = explainer.explain(
                recommended_title=t,
                query_title=movie_title,
                user_id=user_id,
                rec_source='hybrid',
                sentiment_score=s_score,
            )

            final_recs.append(RecommendationItem(
                movieId=m_id,
                title=t,
                score=round(f_score, 4),
                explanation=explanation,
                poster_url=get_poster_url(m_id),
            ))

        return final_recs
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similar", response_model=List[RecommendationItem])
def get_similar(
    movie_title: str = Query(..., description="Target movie title"),
    top_n: int = Query(10, description="Number of similar movies", ge=1, le=50),
):
    """Returns pure content-based similar movies using TF-IDF cosine similarity."""
    try:
        raw_recs = recommender.get_content_similar(movie_title, top_n=top_n)
        if not raw_recs:
            raise HTTPException(
                status_code=404,
                detail=f"Movie '{movie_title}' not found in database."
            )

        final_recs = []
        for rec in raw_recs:
            explanation = explainer.explain(
                recommended_title=rec['title'],
                query_title=movie_title,
                user_id=None,
                rec_source='content',
                sentiment_score=None,
            )
            final_recs.append(RecommendationItem(
                movieId=rec['movieId'],
                title=rec['title'],
                score=round(rec['score'], 4),
                explanation=explanation,
                poster_url=get_poster_url(rec['movieId']),
            ))
        return final_recs
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user-profile", response_model=UserProfile)
def get_user_profile(user_id: int = Query(..., description="User ID")):
    """Generates a JSON digest of a user's taste profile from their rating history."""
    try:
        profile = recommender.get_user_taste_profile(user_id)
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found or has no ratings."
            )

        return UserProfile(**profile)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest):
    """Analyze the sentiment of a review text."""
    try:
        result = reranker.analyze_review(request.text)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
