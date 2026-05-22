"""
CineIQ Data Pipeline
====================
Loads MovieLens 25M + TMDB metadata, merges them into a unified DataFrame,
and saves as an optimized Parquet file for the recommendation engine.

Note: The IMDB 50K Reviews dataset has no movie identifiers (only review + sentiment),
so it is used separately for sentiment model training — NOT merged into the movie table.
"""

import pandas as pd
import numpy as np
import os
import ast
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def _safe_parse_json_list(x, key='name', max_items=None):
    """Safely parse a JSON-like string column and extract values by key."""
    if pd.isna(x) or not str(x).strip():
        return np.nan
    try:
        items = ast.literal_eval(str(x))
        names = [d[key] for d in items if key in d]
        if max_items:
            names = names[:max_items]
        return ', '.join(names) if names else np.nan
    except (ValueError, SyntaxError, TypeError):
        return np.nan


def _extract_director(crew_str):
    """Extract director name from TMDB crew JSON."""
    if pd.isna(crew_str) or not str(crew_str).strip():
        return np.nan
    try:
        for d in ast.literal_eval(str(crew_str)):
            if d.get('job') == 'Director':
                return d['name']
        return np.nan
    except (ValueError, SyntaxError, TypeError):
        return np.nan


def _clean_title(title):
    """Remove year suffix from movie title for fuzzy matching. e.g., 'Toy Story (1995)' -> 'toy story'"""
    if pd.isna(title):
        return ''
    return re.sub(r'\s*\(\d{4}\)\s*$', '', str(title)).lower().strip()


def load_data():
    """
    Loads MovieLens, TMDB datasets from data/raw/,
    merges them into a unified DataFrame, and saves as a Parquet file.
    """
    logger.info("=" * 60)
    logger.info("Starting CineIQ Data Pipeline...")
    logger.info("=" * 60)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # ── 1. Load MovieLens (Core dataset) ──────────────────────────────────
    logger.info("Loading MovieLens dataset...")
    movies_path = os.path.join(RAW_DATA_DIR, 'movies.csv')
    try:
        movies = pd.read_csv(movies_path)
        logger.info(f"  Loaded MovieLens movies: {len(movies)} rows, columns: {movies.columns.tolist()}")
    except FileNotFoundError:
        logger.error(f"MovieLens movies.csv not found in {RAW_DATA_DIR}. Please place it there.")
        return

    # ── 2. Load & Parse TMDB Metadata ─────────────────────────────────────
    logger.info("Loading TMDB metadata...")
    tmdb_movies_path = os.path.join(RAW_DATA_DIR, 'tmdb_5000_movies.csv')
    tmdb_credits_path = os.path.join(RAW_DATA_DIR, 'tmdb_5000_credits.csv')
    tmdb = pd.DataFrame()

    try:
        if os.path.exists(tmdb_movies_path) and os.path.exists(tmdb_credits_path):
            tmdb_movies = pd.read_csv(tmdb_movies_path)
            tmdb_credits = pd.read_csv(tmdb_credits_path)
            logger.info(f"  Loaded TMDB movies ({len(tmdb_movies)}) and credits ({len(tmdb_credits)})")

            # Merge TMDB movies and credits
            # Kaggle TMDB 5000: movies has 'id', credits has 'movie_id'
            if 'id' in tmdb_movies.columns and 'movie_id' in tmdb_credits.columns:
                tmdb = tmdb_movies.merge(tmdb_credits, left_on='id', right_on='movie_id', suffixes=('', '_credits'))
            elif 'title' in tmdb_movies.columns and 'title' in tmdb_credits.columns:
                tmdb = tmdb_movies.merge(tmdb_credits, on='title', suffixes=('', '_credits'))
            else:
                logger.warning("  Cannot determine merge key for TMDB movies+credits.")
                tmdb = tmdb_movies.copy()

            logger.info(f"  Merged TMDB datasets: {len(tmdb)} rows")
        else:
            # Fallback for a single combined file
            tmdb_single = os.path.join(RAW_DATA_DIR, 'tmdb_metadata.csv')
            if os.path.exists(tmdb_single):
                tmdb = pd.read_csv(tmdb_single)
                logger.info(f"  Loaded combined TMDB metadata: {len(tmdb)} rows")
            else:
                logger.warning("  TMDB metadata (tmdb_5000_movies.csv / credits.csv) not found. Proceeding without it.")
    except Exception as e:
        logger.error(f"  Error loading TMDB metadata: {e}")

    # ── 3. Parse TMDB JSON fields ─────────────────────────────────────────
    if not tmdb.empty:
        logger.info("Parsing TMDB JSON metadata columns...")

        # Parse genres from TMDB JSON (e.g. [{"id": 28, "name": "Action"}, ...])
        if 'genres' in tmdb.columns:
            # Check if it's JSON-like or already pipe-separated
            sample = str(tmdb['genres'].dropna().iloc[0]) if not tmdb['genres'].dropna().empty else ''
            if sample.startswith('['):
                tmdb['tmdb_genres'] = tmdb['genres'].apply(lambda x: _safe_parse_json_list(x, 'name'))
            else:
                tmdb['tmdb_genres'] = tmdb['genres']

        # Parse keywords
        if 'keywords' in tmdb.columns:
            tmdb['keywords_parsed'] = tmdb['keywords'].apply(lambda x: _safe_parse_json_list(x, 'name', max_items=5))
        else:
            tmdb['keywords_parsed'] = np.nan

        # Parse cast (top 3 actors)
        if 'cast' in tmdb.columns:
            tmdb['cast_parsed'] = tmdb['cast'].apply(lambda x: _safe_parse_json_list(x, 'name', max_items=3))
        else:
            tmdb['cast_parsed'] = np.nan

        # Parse director from crew
        if 'crew' in tmdb.columns:
            tmdb['director'] = tmdb['crew'].apply(_extract_director)
        else:
            tmdb['director'] = np.nan

        logger.info("  JSON fields parsed successfully.")

    # ── 4. Merge TMDB with MovieLens ──────────────────────────────────────
    if not tmdb.empty:
        logger.info("Merging TMDB metadata with MovieLens...")

        # Create clean title columns for fuzzy matching
        movies['_clean_title'] = movies['title'].apply(_clean_title)

        # Use the TMDB 'title' column (not 'title_credits')
        tmdb_title_col = 'title' if 'title' in tmdb.columns else None
        if tmdb_title_col:
            tmdb['_clean_title'] = tmdb[tmdb_title_col].apply(_clean_title)

            # Select only the columns we need from TMDB to avoid clutter
            tmdb_cols = ['_clean_title']
            col_map = {
                'tmdb_genres': 'tmdb_genres',
                'keywords_parsed': 'keywords',
                'cast_parsed': 'cast',
                'director': 'director',
                'overview': 'overview',
                'vote_average': 'tmdb_vote_avg',
                'popularity': 'tmdb_popularity',
            }

            for src, dst in col_map.items():
                if src in tmdb.columns:
                    tmdb_cols.append(src)

            tmdb_slim = tmdb[tmdb_cols].copy()

            # Rename columns
            rename_map = {k: v for k, v in col_map.items() if k in tmdb_slim.columns and k != v}
            tmdb_slim = tmdb_slim.rename(columns=rename_map)

            # Drop duplicates on clean_title (keep first)
            tmdb_slim = tmdb_slim.drop_duplicates(subset='_clean_title', keep='first')

            # Merge
            before_count = len(movies)
            movies = movies.merge(tmdb_slim, on='_clean_title', how='left')
            matched = movies[movies['cast'].notna()].shape[0] if 'cast' in movies.columns else 0
            logger.info(f"  Merged: {before_count} movies, {matched} matched with TMDB metadata")
        else:
            logger.warning("  TMDB dataframe has no 'title' column for merging.")

        # Drop helper column
        if '_clean_title' in movies.columns:
            movies = movies.drop(columns=['_clean_title'])

    # ── 5. Save to Parquet ────────────────────────────────────────────────
    output_path = os.path.join(PROCESSED_DATA_DIR, 'movies_merged.parquet')
    logger.info(f"Saving merged dataset to {output_path}...")

    try:
        # Ensure all object columns are strings (not lists) for Parquet compatibility
        for col in movies.select_dtypes(include=['object']).columns:
            movies[col] = movies[col].apply(lambda x: str(x) if isinstance(x, list) else x)

        movies.to_parquet(output_path, engine='pyarrow', index=False)
        logger.info(f"Successfully saved unified DataFrame with {len(movies)} records to {output_path}")
        logger.info(f"Final columns: {movies.columns.tolist()}")
    except Exception as e:
        logger.error(f"Failed to save Parquet file: {e}")

    logger.info("=" * 60)
    logger.info("Data Pipeline complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    load_data()
