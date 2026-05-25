"""
CineIQ Streamlit Dashboard
===========================
Professional dark-themed interface for the CineIQ recommendation engine.
"""

import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API_URL = os.environ.get("CINEIQ_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="CineIQ — Movie Recommender",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional Dark Theme CSS ──────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-card: #1c2128;
        --border: #30363d;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --text-muted: #6e7681;
        --accent: #58a6ff;
        --accent-hover: #79c0ff;
        --green: #3fb950;
        --orange: #d29922;
        --red: #f85149;
    }

    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Typography */
    h1 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1.75rem !important;
        letter-spacing: -0.3px;
        border-bottom: 1px solid var(--border);
        padding-bottom: 12px !important;
    }
    h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    p, span, label, .stMarkdown {
        color: var(--text-secondary);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        border-bottom: none !important;
    }

    /* Buttons — all states and inner elements */
    .stButton > button,
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-primary"],
    .stButton > button:focus,
    .stButton > button:active {
        background-color: #1f6feb !important;
        color: #000000 !important;
        border: 1px solid #388bfd !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 8px 20px !important;
    }
    .stButton > button p,
    .stButton > button span,
    .stButton > button div,
    button[data-testid="baseButton-primary"] p,
    button[data-testid="baseButton-primary"] span {
        color: #000000 !important;
    }
    .stButton > button:hover,
    button[data-testid="baseButton-primary"]:hover {
        background-color: #388bfd !important;
        color: #000000 !important;
        border-color: #58a6ff !important;
    }
    .stButton > button:hover p,
    .stButton > button:hover span,
    .stButton > button:hover div {
        color: #000000 !important;
    }

    /* Cards */
    .rec-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .rec-card:hover {
        border-color: var(--accent);
    }
    .rec-title {
        font-size: 15px;
        font-weight: 500;
        color: var(--text-primary);
        margin: 0;
    }
    .rec-rank {
        color: var(--text-muted);
        font-size: 13px;
        font-weight: 400;
        margin-right: 8px;
    }
    .rec-score {
        display: inline-block;
        padding: 2px 8px;
        font-size: 12px;
        font-weight: 500;
        border-radius: 4px;
        background-color: rgba(88, 166, 255, 0.12);
        color: var(--accent);
        border: 1px solid rgba(88, 166, 255, 0.25);
    }
    .rec-explanation {
        font-size: 13px;
        color: var(--text-secondary);
        margin-top: 8px;
        line-height: 1.5;
        padding-left: 2px;
    }

    /* Stat cards */
    .stat-box {
        background-color: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8em;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.2;
    }
    .stat-label {
        font-size: 12px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 4px;
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 6px !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    /* Dividers */
    hr {
        border-color: var(--border) !important;
    }

    /* Selectbox / Radio */
    .stRadio > div {
        gap: 4px;
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    /* Success/Warning/Error messages */
    .stSuccess {
        background-color: rgba(63, 185, 80, 0.08);
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    .stWarning {
        background-color: rgba(210, 153, 34, 0.08);
        border: 1px solid rgba(210, 153, 34, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.markdown("### CineIQ")
st.sidebar.caption("Explainable Movie Recommendations")
st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate",
    ["Recommendations", "Taste Profile", "Similar Movies", "Sentiment"],
    label_visibility="collapsed",
)


# ── API Helpers ───────────────────────────────────────────────────────────
def _get(endpoint, params, timeout=120):
    try:
        r = requests.get(f"{API_URL}{endpoint}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Ensure the FastAPI server is running on port 8000.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def _post(endpoint, json_body, timeout=30):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=json_body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Ensure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def render_card(rank, rec):
    st.markdown(f"""
    <div class="rec-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span class="rec-rank">#{rank}</span>
                <span class="rec-title">{rec['title']}</span>
            </div>
            <span class="rec-score">{rec['score']:.3f}</span>
        </div>
        <div class="rec-explanation">{rec['explanation']}</div>
    </div>
    """, unsafe_allow_html=True)


# Plotly theme helper
def _layout(fig, **kwargs):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8b949e', family='Inter, sans-serif', size=12),
        margin=dict(t=30, b=30, l=20, r=20),
        showlegend=False,
        coloraxis_showscale=False,
        **kwargs,
    )
    fig.update_xaxes(gridcolor='#21262d', zerolinecolor='#21262d')
    fig.update_yaxes(gridcolor='#21262d', zerolinecolor='#21262d')
    return fig


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Recommendations
# ═════════════════════════════════════════════════════════════════════════
if page == "Recommendations":
    st.title("Hybrid Recommendations")
    st.caption("Content-based filtering + collaborative filtering + sentiment re-ranking")
    st.divider()

    st.sidebar.divider()
    st.sidebar.markdown("#### Parameters")
    user_id = st.sidebar.number_input("User ID", min_value=1, value=2, step=1)
    movie_title = st.sidebar.text_input("Seed movie", value="Toy Story (1995)")
    top_n = st.sidebar.slider("Results", 5, 20, 10)

    if st.button("Get recommendations", type="primary"):
        with st.spinner("Computing..."):
            recs = _get("/recommend", {"user_id": user_id, "movie_title": movie_title, "top_n": top_n})
            if recs:
                st.success(f"Showing {len(recs)} results for \"{movie_title}\"")
                for i, rec in enumerate(recs):
                    render_card(i + 1, rec)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Taste Profile
# ═════════════════════════════════════════════════════════════════════════
elif page == "Taste Profile":
    st.title("User Taste Profile")
    st.caption("Genre distribution, decade preferences, and top actor/director affinities from rating history")
    st.divider()

    col_input, _ = st.columns([1, 3])
    with col_input:
        user_id = st.number_input("User ID", min_value=1, value=1)

    if st.button("Load profile", type="primary"):
        with st.spinner("Analysing watch history..."):
            profile = _get("/user-profile", {"user_id": user_id})
            if profile:
                # Stats row
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""<div class="stat-box">
                        <div class="stat-number">{profile.get('total_rated', '—')}</div>
                        <div class="stat-label">Movies Rated</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="stat-box">
                        <div class="stat-number">{profile.get('highly_rated', '—')}</div>
                        <div class="stat-label">Rated 4 stars or above</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    fav = profile['top_genres'][0] if profile.get('top_genres') else '—'
                    st.markdown(f"""<div class="stat-box">
                        <div class="stat-number" style="font-size:1.4em;">{fav}</div>
                        <div class="stat-label">Top Genre</div>
                    </div>""", unsafe_allow_html=True)

                st.divider()
                left, right = st.columns(2)

                # Genre radar
                with left:
                    st.markdown("#### Genre Distribution")
                    genres = profile.get('top_genres', [])
                    if genres:
                        genre_dist = profile.get('genre_distribution', {})
                        vals = [genre_dist.get(g, 1) for g in genres]
                        fig = go.Figure(data=go.Scatterpolar(
                            r=vals + [vals[0]],
                            theta=genres + [genres[0]],
                            fill='toself',
                            fillcolor='rgba(88, 166, 255, 0.1)',
                            line=dict(color='#58a6ff', width=1.5),
                            marker=dict(size=4),
                        ))
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, showticklabels=False, gridcolor='#21262d'),
                                angularaxis=dict(gridcolor='#21262d', color='#8b949e'),
                                bgcolor='rgba(0,0,0,0)',
                            ),
                        )
                        _layout(fig, height=350)
                        st.plotly_chart(fig, use_container_width=True)

                # Decade preferences
                with right:
                    st.markdown("#### Decade Preferences")
                    decades = profile.get('top_decades', [])
                    if decades:
                        df_dec = pd.DataFrame({"Decade": decades, "Weight": list(range(len(decades), 0, -1))})
                        fig = px.bar(df_dec, x="Decade", y="Weight", color="Weight",
                                     color_continuous_scale=["#161b22", "#58a6ff"])
                        _layout(fig, height=350)
                        fig.update_traces(marker_line_width=0)
                        st.plotly_chart(fig, use_container_width=True)

                st.divider()
                a_col, d_col = st.columns(2)

                with a_col:
                    st.markdown("#### Top Actors")
                    actors = profile.get('top_actors', [])
                    if actors and actors[0] != "No cast data available":
                        df_a = pd.DataFrame({"Actor": actors, "Affinity": list(range(len(actors), 0, -1))})
                        fig = px.bar(df_a, x="Affinity", y="Actor", orientation='h',
                                     color="Affinity", color_continuous_scale=["#161b22", "#3fb950"])
                        _layout(fig, height=280)
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No cast metadata available for this user's rated movies.")

                with d_col:
                    st.markdown("#### Top Directors")
                    dirs = profile.get('top_directors', [])
                    if dirs and dirs[0] != "No director data available":
                        df_d = pd.DataFrame({"Director": dirs, "Affinity": list(range(len(dirs), 0, -1))})
                        fig = px.bar(df_d, x="Affinity", y="Director", orientation='h',
                                     color="Affinity", color_continuous_scale=["#161b22", "#d29922"])
                        _layout(fig, height=250)
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No director metadata available for this user's rated movies.")


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Similar Movies
# ═════════════════════════════════════════════════════════════════════════
elif page == "Similar Movies":
    st.title("Content Similarity Search")
    st.caption("TF-IDF cosine similarity on genres, cast, keywords, and plot overview")
    st.divider()

    movie_title = st.text_input("Movie title", "Inception (2010)")

    if st.button("Search", type="primary"):
        with st.spinner("Searching..."):
            recs = _get("/similar", {"movie_title": movie_title, "top_n": 10})
            if recs:
                st.success(f"{len(recs)} similar movies found")
                for i, rec in enumerate(recs):
                    render_card(i + 1, rec)





# ═════════════════════════════════════════════════════════════════════════
# PAGE: Sentiment
# ═════════════════════════════════════════════════════════════════════════
elif page == "Sentiment":
    st.title("Review Sentiment Analysis")
    st.caption("VADER / DistilBERT-powered sentiment scoring on arbitrary review text")
    st.divider()

    review_text = st.text_area(
        "Review text",
        height=180,
        placeholder="Paste a movie review here...",
    )

    if st.button("Analyze", type="primary"):
        if not review_text.strip():
            st.warning("Please enter review text to analyse.")
        else:
            with st.spinner("Analysing..."):
                result = _post("/analyze-sentiment", {"text": review_text})
                if result:
                    score = result['score']
                    label = result['label'].upper()

                    if score >= 0.7:
                        color = "#3fb950"
                    elif score >= 0.4:
                        color = "#d29922"
                    else:
                        color = "#f85149"

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"""<div class="stat-box">
                            <div class="stat-number" style="color:{color}">{score:.2f}</div>
                            <div class="stat-label">Sentiment Score</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""<div class="stat-box">
                            <div class="stat-number" style="color:{color}; font-size:1.4em;">{label}</div>
                            <div class="stat-label">Classification</div>
                        </div>""", unsafe_allow_html=True)

                    # Gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        number=dict(font=dict(color=color)),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': '#30363d'},
                            'bar': {'color': color},
                            'bgcolor': '#161b22',
                            'borderwidth': 1,
                            'bordercolor': '#30363d',
                            'steps': [
                                {'range': [0, 0.4], 'color': 'rgba(248,81,73,0.06)'},
                                {'range': [0.4, 0.7], 'color': 'rgba(210,153,34,0.06)'},
                                {'range': [0.7, 1], 'color': 'rgba(63,185,80,0.06)'},
                            ],
                        },
                    ))
                    _layout(fig, height=240)
                    st.plotly_chart(fig, use_container_width=True)
