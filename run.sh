#!/bin/bash
set -e

# Navigate to project root
cd "$(dirname "$0")"

echo "======================================"
echo "🎬 Starting CineIQ Pipeline..."
echo "======================================"

# 0. Install Dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt --quiet 2>/dev/null || pip3 install -r requirements.txt --quiet 2>/dev/null || {
    echo "⚠️  Could not install dependencies. Make sure pip is available."
    echo "    Continuing anyway..."
}

# 1. Check Datasets
echo ""
echo "📂 Checking for raw datasets in data/raw/..."
MISSING=0
for f in "movies.csv" "ratings.csv"; do
    if [ ! -f "data/raw/$f" ]; then
        echo "   ⚠️ Missing: data/raw/$f"
        MISSING=1
    else
        echo "   ✅ Found: data/raw/$f"
    fi
done

for f in "tmdb_5000_movies.csv" "tmdb_5000_credits.csv"; do
    if [ ! -f "data/raw/$f" ]; then
        echo "   ⚠️ Optional missing: data/raw/$f (TMDB metadata)"
    else
        echo "   ✅ Found: data/raw/$f"
    fi
done

if [ ! -f "data/raw/imdb_reviews.csv" ]; then
    echo "   ⚠️ Missing: data/raw/imdb_reviews.csv (IMDB dataset for sentiment training)"
else
    echo "   ✅ Found: data/raw/imdb_reviews.csv"
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "❌ Required MovieLens datasets are missing. Please download them:"
    echo "   MovieLens 25M: https://grouplens.org/datasets/movielens/25m/"
    echo "   Place movies.csv and ratings.csv in data/raw/"
    exit 1
fi

# 2. Run Preprocessing
echo ""
echo "⚙️  Step 1/5: Running data pipeline (merge datasets)..."
python3 data_loader.py || python data_loader.py

# 3. Train Sentiment Model
echo ""
echo "💬 Step 2/5: Training Sentiment Classifier on IMDB 50K..."
python3 models/train_sentiment.py || python models/train_sentiment.py

# 4. Train Models (using sampled ratings for speed)
echo ""
echo "🧠 Step 3/5: Training Collaborative Filtering Model (SVD)..."
echo "   (This may take a while on the full MovieLens 25M dataset)"
python3 -c "
from models.recommender import HybridRecommender
rec = HybridRecommender()
rec.load_data()
rec.train_collaborative(sample_size=500000)
print('Training complete!')
" || python -c "
from models.recommender import HybridRecommender
rec = HybridRecommender()
rec.load_data()
rec.train_collaborative(sample_size=500000)
print('Training complete!')
"

# 5. Verify the pipeline
echo ""
echo "🔍 Step 4/5: Running quick verification..."
python3 debug_profile.py || python debug_profile.py

# 6. Start Services
echo ""
echo "🚀 Step 5/5: Starting services..."

# Start FastAPI
echo "   Starting FastAPI backend on port 8000..."
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait for API to be ready
sleep 3

# Start Streamlit
echo "   Starting Streamlit dashboard on port 8501..."
python3 -m streamlit run dashboard/app.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!

echo ""
echo "======================================"
echo "🎉 CineIQ is now running!"
echo ""
echo "   🌐 API Docs:   http://localhost:8000/docs"
echo "   📊 Dashboard:  http://localhost:8501"
echo ""
echo "   Press Ctrl+C to stop all services."
echo "======================================"

# Trap Ctrl+C to kill background processes
cleanup() {
    echo ""
    echo "Shutting down CineIQ..."
    kill $FASTAPI_PID 2>/dev/null
    kill $STREAMLIT_PID 2>/dev/null
    echo "Done."
    exit 0
}
trap cleanup INT TERM

# Wait for background processes
wait $FASTAPI_PID $STREAMLIT_PID
