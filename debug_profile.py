"""
CineIQ Debug Profile Script
=============================
Quick diagnostic script to verify user profile generation works correctly.
Tests the complete pipeline: loading data → merging ratings → extracting taste profile.
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.recommender import HybridRecommender


def test_user_profile(user_id=1):
    """Test user profile generation end-to-end."""
    try:
        print(f"Testing user profile for user_id={user_id}...")
        print("=" * 50)

        rec = HybridRecommender()
        if not rec.load_data():
            print("ERROR: Failed to load data.")
            return

        profile = rec.get_user_taste_profile(user_id)
        if profile is None:
            print(f"User {user_id} has no ratings or no highly-rated movies.")
            return

        print(f"✅ User {user_id} Profile Generated Successfully!")
        print(f"   Total Rated: {profile['total_rated']}")
        print(f"   Highly Rated (4★+): {profile['highly_rated']}")
        print(f"   Top Genres: {profile['top_genres']}")
        print(f"   Top Decades: {profile['top_decades']}")
        print(f"   Top Actors: {profile['top_actors']}")
        print(f"   Top Directors: {profile['top_directors']}")
        print()

        # Also test content similarity
        print("Testing content similarity for 'Toy Story (1995)'...")
        results = rec.get_content_similar("Toy Story (1995)", top_n=5)
        if results:
            for r in results:
                print(f"   → {r['title']} (score: {r['score']:.4f})")
        else:
            print("   No results found.")

        print()
        print("✅ All tests passed. No 500 error expected.")

    except Exception as e:
        print(f"❌ ERROR OCCURRED: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    user_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    test_user_profile(user_id)
