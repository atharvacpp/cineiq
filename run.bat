@echo off
cd /d "%~dp0"
echo ======================================
echo 🎬 Starting CineIQ Pipeline...
echo ======================================

echo.
echo 📦 Installing Python dependencies...
python -m pip install -r requirements.txt --quiet

echo.
echo ⚙️  Step 1/5: Running data pipeline...
python data_loader.py

echo.
echo 💬 Step 2/5: Training Sentiment Classifier on IMDB 50K...
python models/train_sentiment.py

echo.
echo 🧠 Step 3/5: Training SVD model...
python -c "from models.recommender import HybridRecommender; rec = HybridRecommender(); rec.load_data(); rec.train_collaborative(sample_size=500000)"

echo.
echo 🔍 Step 4/5: Running verification...
python debug_profile.py

echo.
echo 🚀 Step 5/5: Starting services...

echo Starting FastAPI backend on port 8000 in a new window...
start cmd /k "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

echo Starting Streamlit dashboard on port 8501 in a new window...
start cmd /k "python -m streamlit run dashboard/app.py --server.port 8501"

echo.
echo ======================================
echo 🎉 CineIQ is launching!
echo.
echo    API Docs:  http://localhost:8000/docs
echo    Dashboard: http://localhost:8501
echo ======================================
pause
