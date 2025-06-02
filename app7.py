import streamlit as st
import pandas as pd
import pickle
import requests
import os
import csv
from datetime import datetime
import uuid
import numpy as np
import scipy.sparse as sp
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import bcrypt

# TMDB API Key (use environment variable for security)
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "9ef5ae6fc8b8f484e9295dc97d8d32ea")

# Load .pkl files
try:
    movies = pickle.load(open('movie_list.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    svd_model = pickle.load(open('svd_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Could not find 'movie_list.pkl', 'similarity.pkl', or 'svd_model.pkl'. .")
    movies = pd.DataFrame()
    similarity = None
    svd_model = None

# Custom CSS for dark theme, styling, and watchlist/history cards
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .stApp {
        background-color: #121212;
        color: white;
    }
    h1, h2, h3 {
        color: white;
    }
    .stButton>button {
        background-color: #4a4a4a;
        color: white;
        border-radius: 5px;
    }
    .stTextInput>div>input, .stSelectbox>div>select {
        background-color: #2a2a2a;
        color: white;
        border-radius: 5px;
    }
    .genre-button {
        background-color: #2a2a2a;
        color: white;
        border-radius: 15px;
        padding: 5px 15px;
        margin: 5px;
        display: inline-block;
    }
    .movie-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
        width: 200px;
        display: inline-block;
        vertical-align: top;
    }
    .watchlist-container, .history-container {
        display: flex;
        overflow-x: auto;
        white-space: nowrap;
        padding: 10px 0;
    }
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px;
        background-color: #1a1a1a;
    }
    .nav-links button {
        background: none;
        border: none;
        color: white;
        margin: 0 15px;
        text-decoration: none;
        cursor: pointer;
    }
    .sign-in-btn {
        background-color: #333;
        color: white;
        padding: 5px 15px;
        border-radius: 5px;
        text-decoration: none;
        border: none;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# Cache TMDB API calls for performance
@st.cache_data
def fetch_popular_movies():
    url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&language=en-US&page=1"
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[204, 429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.get(url, timeout=5)
        if response.status_code != 200:
            st.warning(f"Failed to fetch popular movies: HTTP {response.status_code}")
            return []
        data = response.json()
        if not isinstance(data, dict) or "results" not in data:
            st.warning("Invalid popular movies response")
            return []
        movies_list = []
        for movie in data.get("results", []):
            movies_list.append({
                "id": movie.get("id", 0),
                "title": movie.get("title", "Unknown"),
                "rating": movie.get("vote_average", 0.0),
                "description": movie.get("overview", "No description available"),
                "poster": f"https://image.tmdb.org/t/p/w500/{movie.get('poster_path')}" if movie.get("poster_path") else "https://via.placeholder.com/200x300?text=No+Poster",
                "runtime": movie.get("runtime", 120),
                "release_date": movie.get("release_date", "2000-01-01"),
                "genres": movie.get("genre_ids", [])
            })
        return movies_list
    except requests.exceptions.RequestException as e:
        st.warning(f"Network error fetching popular movies: {e}")
        return []
    except Exception as e:
        st.warning(f"Error fetching popular movies: {e}")
        return []

@st.cache_data
def fetch_genres():
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[204, 429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.get(url, timeout=5)
        if response.status_code != 200:
            st.warning(f"Failed to fetch genres: HTTP {response.status_code}")
            return {}
        data = response.json()
        if not isinstance(data, dict) or "genres" not in data:
            st.warning("Invalid genres response")
            return {}
        return {genre["id"]: genre["name"] for genre in data.get("genres", [])}
    except Exception as e:
        st.warning(f"Error fetching genres: {e}")
        return {}

@st.cache_data
def fetch_movies_by_genre(genre_id):
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_genres={genre_id}&language=en-US&page=1"
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[204, 429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.get(url, timeout=5)
        if response.status_code != 200:
            st.warning(f"Failed to fetch movies for genre: HTTP {response.status_code}")
            return []
        data = response.json()
        if not isinstance(data, dict) or "results" not in data:
            st.warning("Invalid genre movies response")
            return []
        movies_list = []
        for movie in data.get("results", []):
            movies_list.append({
                "id": movie.get("id", 0),
                "title": movie.get("title", "Unknown"),
                "rating": movie.get("vote_average", 0.0),
                "description": movie.get("overview", "No description available"),
                "poster": f"https://image.tmdb.org/t/p/w500/{movie.get('poster_path')}" if movie.get("poster_path") else "https://via.placeholder.com/200x300?text=No+Poster",
                "runtime": movie.get("runtime", 120),
                "release_date": movie.get("release_date", "2000-01-01"),
                "genres": movie.get("genre_ids", [])
            })
        return movies_list
    except Exception as e:
        st.warning(f"Error fetching movies for genre: {e}")
        return []

@st.cache_data
def fetch_poster(movie_id):
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[204, 429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = session.get(url, timeout=5)
        
        if response.status_code == 204:
            st.warning(f"No poster available for movie ID {movie_id} (HTTP 204)")
            return "https://via.placeholder.com/200x300?text=No+Poster"
        elif response.status_code != 200:
            st.warning(f"Failed to fetch poster for movie ID {movie_id}: HTTP {response.status_code}")
            return "https://via.placeholder.com/200x300?text=Error"
        
        data = response.json()
        if not isinstance(data, dict):
            st.warning(f"Invalid poster response for movie ID {movie_id}")
            return "https://via.placeholder.com/200x300?text=Error"
        
        poster_path = data.get('poster_path')
        return f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else "https://via.placeholder.com/200x300?text=No+Poster"
    except requests.exceptions.ConnectionError as e:
        st.warning(f"Network error fetching poster for movie ID {movie_id}: {e}")
        return "https://via.placeholder.com/200x300?text=Network+Error"
    except requests.exceptions.Timeout:
        st.warning(f"Request timed out fetching poster for movie ID {movie_id}")
        return "https://via.placeholder.com/200x300?text=Timeout"
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching poster for movie ID {movie_id}: {e}")
        return "https://via.placeholder.com/200x300?text=Error"
    except Exception as e:
        st.warning(f"Unexpected error fetching poster for movie ID {movie_id}: {e}")
        return "https://via.placeholder.com/200x300?text=Error"

@st.cache_data
def fetch_trailer(movie_id):
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[204, 429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}&language=en-US"
        response = session.get(url, timeout=5)
        
        if response.status_code == 204:
            st.warning(f"No trailer available for movie ID {movie_id} (HTTP 204)")
            return None
        elif response.status_code != 200:
            st.warning(f"Failed to fetch trailer for movie ID {movie_id}: HTTP {response.status_code}")
            return None
        
        data = response.json()
        if not isinstance(data, dict) or "results" not in data:
            st.warning(f"Invalid trailer response for movie ID {movie_id}")
            return None
        
        for video in data.get('results', []):
            if video.get('type') == 'Trailer' and video.get('site') == 'YouTube':
                return f"https://www.youtube.com/watch?v={video['key']}"
        return None
    except requests.exceptions.ConnectionError as e:
        st.warning(f"Network error fetching trailer for movie ID {movie_id}. Please check internet connection.")
        return None
    except requests.exceptions.Timeout:
        st.warning(f"Request timed out fetching trailer for movie ID {movie_id}. Please try again later.")
        return None
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching trailer for movie ID {movie_id}: {e}")
        return None
    except Exception as e:
        st.warning(f"Unexpected error fetching trailer for movie ID {movie_id}: {e}")
        return None

@st.cache_data
def fetch_movie_details(movie_id):
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[204, 429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = session.get(url, timeout=5)
        
        if response.status_code == 204:
            st.warning(f"No details available for movie ID {movie_id} (HTTP 204)")
            return {"rating": 0.0, "description": "No description available"}
        elif response.status_code != 200:
            st.warning(f"Failed to fetch details for movie ID {movie_id}: HTTP {response.status_code}")
            return {"rating": 0.0, "description": "No description available"}
        
        data = response.json()
        if not isinstance(data, dict):
            st.warning(f"Invalid details response for movie ID {movie_id}")
            return {"rating": 0.0, "description": "No description available"}
        
        return {
            "rating": data.get("vote_average", 0.0),
            "description": data.get("overview", "No description available")
        }
    except Exception as e:
        st.warning(f"Error fetching movie details for movie ID {movie_id}: {e}")
        return {"rating": 0.0, "description": "No description available"}

@st.cache_data
def fetch_movie_metadata(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=keywords"
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[204, 429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.get(url, timeout=5)
        if response.status_code != 200:
            st.warning(f"Failed to fetch metadata for movie ID {movie_id}: HTTP {response.status_code}")
            return {"genres": [], "keywords": [], "title": "Unknown"}
        data = response.json()
        return {
            "genres": [g['id'] for g in data.get('genres', [])],
            "keywords": [k['id'] for k in data.get('keywords', {}).get('keywords', [])[:5]],
            "title": data.get('title', 'Unknown'),
            "rating": data.get('vote_average', 0.0),
            "description": data.get('overview', 'No description available')
        }
    except Exception as e:
        st.warning(f"Error fetching metadata for movie ID {movie_id}: {e}")
        return {"genres": [], "keywords": [], "title": "Unknown", "rating": 0.0, "description": "No description available"}

@st.cache_data
def fetch_mood_based_movies(_cache_key, genre_ids, max_runtime=None, min_year=None, max_year=None, keywords=None, adult=False):
    movies_list = []
    base_url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&language=en-US&sort_by=vote_average.desc&vote_count.gte=100"
    
    # Construct query parameters
    query_params = []
    if genre_ids:
        query_params.append(f"with_genres={','.join(map(str, genre_ids))}")
    query_params.append(f"include_adult={adult}")
    
    # Try multiple query variations for diversity
    attempts = [
        # Full query
        query_params + (
            ([f"with_runtime.lte={max_runtime}"] if max_runtime else []) +
            ([f"primary_release_date.gte={min_year}-01-01"] if min_year else []) +
            ([f"primary_release_date.lte={max_year}-12-31"] if max_year else []) +
            ([f"with_keywords={keywords}"] if keywords else []) +
            ["page=1"]
        ),
        # Relax runtime and keywords
        query_params + (
            ([f"primary_release_date.gte={min_year}-01-01"] if min_year else []) +
            ([f"primary_release_date.lte={max_year}-12-31"] if max_year else []) +
            ["page=1"]
        ),
        # Random page for diversity
        query_params + (
            ([f"primary_release_date.gte={min_year}-01-01"] if min_year else []) +
            ([f"primary_release_date.lte={max_year}-12-31"] if max_year else []) +
            [f"page={np.random.randint(1, 5)}"]
        ),
        # Broad query
        query_params + ["page=1"],
    ]
    
    for params in attempts:
        url = base_url + "&" + "&".join(params)
        try:
            session = requests.Session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[204, 429, 500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retries))
            response = session.get(url, timeout=5)
            if response.status_code != 200:
                continue
            data = response.json()
            if not isinstance(data, dict) or "results" not in data:
                continue
            for movie in data.get("results", [])[:5]:
                movies_list.append({
                    "id": movie.get("id", 0),
                    "title": movie.get("title", "Unknown"),
                    "rating": movie.get("vote_average", 0.0),
                    "description": movie.get("overview", "No description available"),
                    "poster": f"https://image.tmdb.org/t/p/w500/{movie.get('poster_path')}" if movie.get("poster_path") else "https://via.placeholder.com/200x300?text=No+Poster",
                    "runtime": movie.get("runtime", 120),
                    "release_date": movie.get("release_date", "2000-01-01"),
                    "genres": movie.get("genre_ids", [])
                })
            if movies_list:
                # Shuffle for diversity
                np.random.shuffle(movies_list)
                return movies_list[:5]
        except Exception as e:
            continue
    
    # Fallback to popular movies with warning
    st.warning("No movies found matching your mood-based criteria. Showing popular movies.")
    return fetch_popular_movies()

# Content-based recommendation
def recommend_content_based(movie_title):
    if similarity is None or movies.empty:
        st.error("Content-based recommendations unavailable due to missing data.")
        return [], []
    try:
        index = movies[movies['title'] == movie_title].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_names = []
        recommended_posters = []
        for i in distances[1:6]:
            movie_id = movies.iloc[i[0]].id
            recommended_names.append(movies.iloc[i[0]].title)
            recommended_posters.append(fetch_poster(movie_id))
        return recommended_names, recommended_posters
    except IndexError:
        st.error(f"Movie '{movie_title}' not found in the database.")
        return [], []
    except Exception as e:
        st.error(f"Error generating content-based recommendations: {e}")
        return [], []

# Fallback content-based recommendation using TMDB genres
def recommend_content_based_tmdb(movie_title, num_recommendations=5):
    if movies.empty:
        st.error("Content-based recommendations unavailable due to missing movie data.")
        return [], []
    
    try:
        movie_id = movies[movies['title'] == movie_title]['id'].iloc[0] if movie_title in movies['title'].values else None
        if not movie_id:
            st.error(f"Movie '{movie_title}' not found.")
            return [], []
        
        # Fetch metadata for the target movie
        target_metadata = fetch_movie_metadata(movie_id)
        target_genres = set(target_metadata['genres'])
        
        # Compute similarities with other movies
        similarities = []
        for movie in movies.itertuples():
            if movie.title == movie_title:
                continue
            metadata = fetch_movie_metadata(movie.id)
            genres = set(metadata['genres'])
            # Jaccard similarity for genres
            intersection = len(target_genres & genres)
            union = len(target_genres | genres)
            sim = intersection / union if union > 0 else 0
            similarities.append((movie.id, movie.title, sim))
        
        # Sort by similarity and select top recommendations
        similarities = sorted(similarities, key=lambda x: x[2], reverse=True)[:num_recommendations]
        recommended_names = [title for _, title, _ in similarities]
        recommended_posters = [fetch_poster(movie_id) for movie_id, _, _ in similarities]
        return recommended_names, recommended_posters
    
    except IndexError:
        st.error(f"Movie '{movie_title}' not found in the database.")
        return [], []
    except Exception as e:
        st.error(f"Error generating TMDB-based content recommendations: {e}")
        return [], []

# Collaborative filtering recommendation (using SVD)
def recommend_collaborative(user_id):
    if svd_model is None or movies.empty:
        st.error("Collaborative recommendations unavailable due to missing data.")
        return [], []
    try:
        recommended_names = []
        recommended_posters = []
        predictions = [(movie_id, svd_model.predict(user_id, movie_id).est) for movie_id in movies['id']]
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        for movie_id, _ in predictions[:3]:
            movie_title = movies[movies['id'] == movie_id]['title'].iloc[0]
            recommended_names.append(movie_title)
            recommended_posters.append(fetch_poster(movie_id))
        return recommended_names, recommended_posters
    except Exception as e:
        st.error(f"Error generating collaborative recommendations: {e}")
        return [], []

# Hybrid recommendation (combining content-based and SVD-based collaborative filtering)
def recommend_hybrid(movie_title, user_id, num_recommendations=3, content_weight=0.5):
    if movies.empty:
        st.error("Hybrid recommendations unavailable due to missing movie data.")
        return [], []
    
    try:
        # Get content-based recommendations
        if similarity is not None:
            content_names, content_posters = recommend_content_based(movie_title)
        else:
            content_names, content_posters = recommend_content_based_tmdb(movie_title, num_recommendations * 2)
        content_scores = {name: score for name, score in zip(content_names, np.linspace(1.0, 0.5, len(content_names)))}
        
        # Get collaborative recommendations (SVD-based)
        collab_names, collab_posters = recommend_collaborative(user_id)
        collab_scores = {name: score for name, score in zip(collab_names, np.linspace(1.0, 0.5, len(collab_names)))}
        
        # Combine scores
        combined_scores = {}
        all_names = set(content_names) | set(collab_names)
        
        for name in all_names:
            content_score = content_scores.get(name, 0.0) * content_weight
            collab_score = collab_scores.get(name, 0.0) * (1.0 - content_weight)
            combined_scores[name] = content_score + collab_score
        
        # Sort by combined score
        top_movies = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
        recommended_names = []
        recommended_posters = []
        
        for name, _ in top_movies:
            if name in content_names:
                idx = content_names.index(name)
                recommended_names.append(name)
                recommended_posters.append(content_posters[idx])
            elif name in collab_names:
                idx = collab_names.index(name)
                recommended_names.append(name)
                recommended_posters.append(collab_posters[idx])
            else:
                movie_id = movies[movies['title'] == name]['id'].iloc[0] if name in movies['title'].values else None
                if movie_id:
                    recommended_names.append(name)
                    recommended_posters.append(fetch_poster(movie_id))
        
        if not recommended_names:
            st.warning("No hybrid recommendations found. Falling back to mood-based or popular movies.")
            if st.session_state.mood_answers:
                movies_list = recommend_mood_based(st.session_state.mood_answers, fetch_genres())
                return [m['title'] for m in movies_list[:num_recommendations]], [m['poster'] for m in movies_list[:num_recommendations]]
            popular_movies = fetch_popular_movies()
            return [m['title'] for m in popular_movies[:num_recommendations]], [m['poster'] for m in popular_movies[:num_recommendations]]
        
        return recommended_names, recommended_posters
    
    except IndexError:
        st.error(f"Movie '{movie_title}' not found in the database.")
        popular_movies = fetch_popular_movies()
        return [m['title'] for m in popular_movies[:num_recommendations]], [m['poster'] for m in popular_movies[:num_recommendations]]
    except Exception as e:
        st.error(f"Error generating hybrid recommendations: {e}")
        popular_movies = fetch_popular_movies()
        return [m['title'] for m in popular_movies[:num_recommendations]], [m['poster'] for m in popular_movies[:num_recommendations]]

# Mood-based recommendation
def recommend_mood_based(answers, genre_map):
    genre_ids = []
    max_runtime = None
    min_year = None
    max_year = None
    keywords = None
    adult = False

    mood_genres = {
        "Happy": [35, 16, 12],  # Comedy, Animation, Adventure
        "Sad": [18, 10749, 99],  # Drama, Romance, Documentary
        "Stressed": [35, 10749, 10751],  # Comedy, Romance, Family
        "Excited": [28, 53, 878],  # Action, Thriller, Sci-Fi
        "Relaxed": [18, 10749, 99],  # Drama, Romance, Documentary
        "Bored": [28, 35, 12],  # Action, Comedy, Adventure
        "Angry": [53, 28, 18]  # Thriller, Action, Drama
    }
    secondary_genres = {
        "Happy": [10751],  # Family
        "Sad": [36],  # History
        "Stressed": [16],  # Animation
        "Excited": [12],  # Adventure
        "Relaxed": [35],  # Comedy
        "Bored": [878],  # Sci-Fi
        "Angry": [80]  # Crime
    }
    
    if answers.get("mood"):
        genre_ids.extend(mood_genres.get(answers["mood"], []))
        genre_ids.extend(secondary_genres.get(answers["mood"], []))
    
    if answers.get("motivation") == "Yes":
        genre_ids.extend([18, 99])  # Drama, Documentary
        keywords = "inspirational,motivational"
    
    if answers.get("watching_with") in ["Kids", "Family"] or answers.get("occasion") == "Family Night":
        genre_ids.extend([16, 10751])  # Animation, Family
        adult = False
    elif answers.get("occasion") == "Date Night" or answers.get("romantic") == "Yes":
        genre_ids.extend([10749, 35])  # Romance, Comedy
    
    if answers.get("time"):
        if answers["time"] == "Less than 1 hour":
            max_runtime = 90
        elif answers["time"] == "1-2 hours":
            max_runtime = 120
        elif answers["time"] == "2+ hours":
            max_runtime = 180
    
    if answers.get("genre"):
        genre_id = [k for k, v in genre_map.items() if v == answers["genre"]]
        if genre_id:
            genre_ids.append(genre_id[0])
    
    tone_genres = {
        "Light-hearted": [35, 10749],  # Comedy, Romance
        "Serious": [18, 36],  # Drama, History
        "Emotional": [18, 10749],  # Drama, Romance
        "Fun": [35, 12],  # Comedy, Adventure
        "Epic": [12, 28],  # Adventure, Action
        "Thought-provoking": [18, 99]  # Drama, Documentary
    }
    if answers.get("tone"):
        genre_ids.extend(tone_genres.get(answers["tone"], []))
    
    if answers.get("release"):
        if answers["release"] == "New (post-2010)":
            min_year = 2010
        elif answers["release"] == "Classics (pre-2010)":
            max_year = 2010
    
    if answers.get("mature") == "No":
        adult = False
    elif answers.get("mature") == "Yes":
        adult = True
    
    # Remove duplicates and limit genres
    genre_ids = list(set(genre_ids))[:3]
    
    # Fallback genres if none selected
    if not genre_ids:
        genre_ids = [35, 18]  # Comedy, Drama
    
    # Unique cache key for diversity
    cache_key = str(uuid.uuid4()) + str(answers)
    
    return fetch_mood_based_movies(cache_key, genre_ids, max_runtime, min_year, max_year, keywords, adult)

# Save user activity
def save_user_activity(user_id, action, movie_title, movie_id, rating=None):
    try:
        file_exists = os.path.exists("user_activity.csv")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        activity_data = pd.DataFrame([[user_id, action, movie_title, movie_id, rating, timestamp]], 
                                    columns=["user_id", "action", "title", "movie_id", "rating", "timestamp"])
        activity_data.to_csv("user_activity.csv", mode='a', index=False, header=not file_exists, quoting=csv.QUOTE_NONNUMERIC)
    except Exception as e:
        st.warning(f"Error saving user activity: {e}")

# Save user to CSV
def save_user_to_csv(username, password, user_id):
    try:
        file_exists = os.path.exists("users.csv")
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        user_data = pd.DataFrame([{
            "username": str(username),
            "password": hashed_password,
            "user_id": int(user_id)
        }])
        user_data.to_csv("users.csv", mode='a', index=False, header=not file_exists, quoting=csv.QUOTE_NONNUMERIC)
    except Exception as e:
        st.error(f"Error saving user to CSV: {e}")

# Load users from CSV
def load_users_from_csv():
    if os.path.exists("users.csv"):
        try:
            users_df = pd.read_csv(
                "users.csv",
                dtype={"username": str, "password": str, "user_id": int},
                quoting=csv.QUOTE_NONNUMERIC
            )
            users = {}
            for _, row in users_df.iterrows():
                users[row['username']] = {"password": row['password'], "user_id": row['user_id']}
            return users
        except Exception as e:
            st.warning(f"Error loading users: {e}. Starting with empty user list.")
            return {}
    return {}

# Save watchlist to CSV
def save_watchlist_to_csv(user_id, movie_title, movie_id):
    filename = f"watchlist_{user_id}.csv"
    file_exists = os.path.exists(filename)
    try:
        movie_id = int(movie_id)
        watchlist_data = pd.DataFrame([[movie_title, movie_id]], columns=["title", "movie_id"])
        watchlist_data.to_csv(filename, mode='a', index=False, header=not file_exists, quoting=csv.QUOTE_NONNUMERIC)
    except Exception as e:
        st.error(f"Error saving watchlist for user {user_id}: {e}")

# Load watchlist from CSV
def load_watchlist_from_csv(user_id):
    filename = f"watchlist_{user_id}.csv"
    if os.path.exists(filename):
        try:
            watchlist_df = pd.read_csv(
                filename,
                names=["title", "movie_id"],
                skiprows=1 if os.path.getsize(filename) > 0 else 0,
                on_bad_lines='skip',
                quoting=csv.QUOTE_NONNUMERIC
            )
            watchlist_df = watchlist_df.dropna(subset=["title", "movie_id"])
            watchlist_df["movie_id"] = pd.to_numeric(watchlist_df["movie_id"], errors='coerce', downcast='integer')
            watchlist_df = watchlist_df.dropna(subset=["movie_id"])
            return watchlist_df[["title", "movie_id"]].to_dict('records')
        except Exception as e:
            st.warning(f"Error loading watchlist for user {user_id}: {e}")
            try:
                corrupted_filename = f"watchlist_{user_id}_corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                os.rename(filename, corrupted_filename)
                st.info(f"Corrupted watchlist file renamed to {corrupted_filename}. Starting with an empty watchlist.")
            except Exception as rename_e:
                st.error(f"Failed to rename corrupted watchlist file: {rename_e}")
            return []
    return []

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"
if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = None
if "show_recommendations" not in st.session_state:
    st.session_state.show_recommendations = False
if "recommendation_type" not in st.session_state:
    st.session_state.recommendation_type = None
if "genre_movies" not in st.session_state:
    st.session_state.genre_movies = []
if "users" not in st.session_state:
    st.session_state.users = load_users_from_csv()
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "current_username" not in st.session_state:
    st.session_state.current_username = None
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "mood_answers" not in st.session_state:
    st.session_state.mood_answers = {}
if "mood_recommendations" not in st.session_state:
    st.session_state.mood_recommendations = []

# Load watchlist for the current user
if st.session_state.current_user and not st.session_state.watchlist:
    st.session_state.watchlist = load_watchlist_from_csv(st.session_state.current_user)

# Navigation Bar
st.markdown("<div class='nav-bar'>", unsafe_allow_html=True)
col_logo, col_links, col_signin = st.columns([1, 4, 1])
with col_logo:
    st.markdown("<div style='font-size: 24px; color: #00b7eb;'>🎬 MovieMind</div>", unsafe_allow_html=True)
with col_links:
    col_home, col_discover, col_mood, col_watchlist, col_history = st.columns(5)
    with col_home:
        if st.button("Home", key="nav_home"):
            st.session_state.page = "home"
            st.session_state.show_recommendations = False
            st.session_state.selected_genre = None
    with col_discover:
        if st.button("Discover", key="nav_discover"):
            st.session_state.page = "discover"
    with col_mood:
        if st.button("Mood-Based", key="nav_mood"):
            st.session_state.page = "mood"
    with col_watchlist:
        if st.button("Watchlist", key="nav_watchlist"):
            st.session_state.page = "watchlist"
    with col_history:
        if st.button("History", key="nav_history"):
            st.session_state.page = "history"
with col_signin:
    if st.session_state.current_user:
        if st.button(f"Sign Out ({st.session_state.current_username})", key="nav_signout"):
            st.session_state.current_user = None
            st.session_state.current_username = None
            st.session_state.watchlist = []
            st.session_state.mood_answers = {}
            st.session_state.mood_recommendations = []
            st.session_state.page = "home"
            st.success("Signed out successfully!")
    else:
        if st.button("Sign In", key="nav_signin"):
            st.session_state.page = "signin"
st.markdown("</div>", unsafe_allow_html=True)

# Page Content
if st.session_state.page == "home":
    st.markdown("<h1 style='text-align: center;'>Discover Movies You'll Love</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #b0b0b0;'>Let our AI find your perfect next watch based on your unique taste</p>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center;'>Popular Movies</h2>", unsafe_allow_html=True)
    popular_movies = fetch_popular_movies()
    if popular_movies:
        cols = st.columns(3)
        for idx, movie in enumerate(popular_movies[:3]):
            with cols[idx % 3]:
                trailer_url = fetch_trailer(movie['id'])
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{movie['poster']}" style="width: 100%; border-radius: 10px;">
                        <h3>{movie['title']}</h3>
                        <p>⭐ {movie['rating']:.1f}</p>
                        <p>{movie['description'][:100]}...</p>
                    </div>
                """, unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Watch Now", key=f"watch_pop_{movie['id']}"):
                        if st.session_state.current_user:
                            save_user_activity(st.session_state.current_user, "watched", movie['title'], movie['id'])
                            st.session_state[f"rating_movie_{movie['id']}"] = movie['title']
                            st.session_state[f"show_rating_{movie['id']}"] = True
                            st.session_state[f"rating_movie_id_{movie['id']}"] = movie['id']
                        else:
                            st.warning("Please sign in to watch movies.")
                with col2:
                    if st.button("Add to Watchlist", key=f"watchlist_pop_{movie['id']}"):
                        if st.session_state.current_user:
                            if not any(item["title"] == movie['title'] for item in st.session_state.watchlist):
                                st.session_state.watchlist.append({"title": movie['title'], "movie_id": movie['id']})
                                save_watchlist_to_csv(st.session_state.current_user, movie['title'], movie['id'])
                                save_user_activity(st.session_state.current_user, "added_to_watchlist", movie['title'], movie['id'])
                                st.success(f"Added {movie['title']} to watchlist!")
                        else:
                            st.warning("Please sign in to add movies to your watchlist.")
                with col3:
                    if trailer_url:
                        if st.button("Watch Trailer", key=f"trailer_pop_{movie['id']}"):
                            st.write(f"[Watch Trailer]({trailer_url})")
                if st.session_state.get(f"show_rating_{movie['id']}", False):
                    rating = st.slider(f"Rate {movie['title']} (1-5)", 1, 5, key=f"rating_pop_{movie['id']}")
                    if st.button("Submit Rating", key=f"submit_rating_pop_{movie['id']}"):
                        if st.session_state.current_user:
                            save_user_activity(st.session_state.current_user, "rated", movie['title'], movie['id'], rating)
                            st.success(f"Rated {movie['title']} with {rating} stars!")
                            st.session_state[f"show_rating_{movie['id']}"] = False
                        else:
                            st.warning("Please sign in to rate movies.")
    else:
        st.info("Could not fetch popular movies. Please check your API key or internet connection.")

elif st.session_state.page == "discover":
    if not movies.empty:
        user_id = st.session_state.current_user if st.session_state.current_user else 1
        movie_list = movies['title'].values
        selected_movie = st.selectbox("🎥 Pick a movie for recommendations", movie_list)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Get Content-Based Recommendations", key="content_based"):
                recommended_names, recommended_posters = recommend_content_based(selected_movie)
                if recommended_names:
                    st.session_state.show_recommendations = True
                    st.session_state.recommended_names = recommended_names
                    st.session_state.recommended_posters = recommended_posters
                    st.session_state.recommendation_type = "content"
                else:
                    st.error("Could not generate content-based recommendations.")
        with col2:
            if st.button("Get Collaborative Recommendations", key="collaborative"):
                recommended_names, recommended_posters = recommend_collaborative(user_id)
                if recommended_names:
                    st.session_state.show_recommendations = True
                    st.session_state.recommended_names = recommended_names
                    st.session_state.recommended_posters = recommended_posters
                    st.session_state.recommendation_type = "collaborative"
                else:
                    st.error("Could not generate collaborative recommendations.")
        with col3:
            if st.button("Get Hybrid Recommendations", key="hybrid"):
                recommended_names, recommended_posters = recommend_hybrid(selected_movie, user_id)
                if recommended_names:
                    st.session_state.show_recommendations = True
                    st.session_state.recommended_names = recommended_names
                    st.session_state.recommended_posters = recommended_posters
                    st.session_state.recommendation_type = "hybrid"
                else:
                    st.error("Could not generate hybrid recommendations.")
    else:
        st.warning("No movies loaded from .pkl file. Please ensure the file is correct.")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center;'>Recommended for You</h2>", unsafe_allow_html=True)

    search_query = st.text_input("", placeholder="Search movies...")

    genre_map = fetch_genres()
    genre_names = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance", "Thriller", "Adventure"]
    genre_ids = {name: gid for gid, name in genre_map.items() if name in genre_names}

    cols = st.columns(len(genre_names))
    for idx, genre in enumerate(genre_names):
        with cols[idx]:
            if st.button(genre, key=f"genre_{genre}"):
                genre_id = genre_ids.get(genre)
                if genre_id:
                    st.session_state.selected_genre = genre
                    st.session_state.genre_movies = fetch_movies_by_genre(genre_id)
                    st.session_state.show_recommendations = False

    if st.session_state.selected_genre and st.session_state.genre_movies:
        st.subheader(f"{st.session_state.selected_genre} Movies")
        cols = st.columns(3)
        for idx, movie in enumerate(st.session_state.genre_movies[:3]):
            with cols[idx % 3]:
                trailer_url = fetch_trailer(movie['id'])
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{movie['poster']}" style="width: 100%; border-radius: 10px;">
                        <h3>{movie['title']}</h3>
                        <p>⭐ {movie['rating']:.1f}</p>
                        <p>{movie['description'][:100]}...</p>
                    </div>
                """, unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Watch Now", key=f"watch_genre_{movie['id']}"):
                        if st.session_state.current_user:
                            save_user_activity(st.session_state.current_user, "watched", movie['title'], movie['id'])
                            st.session_state[f"rating_movie_{movie['id']}"] = movie['title']
                            st.session_state[f"show_rating_{movie['id']}"] = True
                            st.session_state[f"rating_movie_id_{movie['id']}"] = movie['id']
                        else:
                            st.warning("Please sign in to watch movies.")
                with col2:
                    if st.button("Add to Watchlist", key=f"watchlist_genre_{movie['id']}"):
                        if st.session_state.current_user:
                            if not any(item["title"] == movie['title'] for item in st.session_state.watchlist):
                                st.session_state.watchlist.append({"title": movie['title'], "movie_id": movie['id']})
                                save_watchlist_to_csv(st.session_state.current_user, movie['title'], movie['id'])
                                save_user_activity(st.session_state.current_user, "added_to_watchlist", movie['title'], movie['id'])
                                st.success(f"Added {movie['title']} to watchlist!")
                        else:
                            st.warning("Please sign in to add movies to your watchlist.")
                with col3:
                    if trailer_url:
                        if st.button("Watch Trailer", key=f"trailer_genre_{movie['id']}"):
                            st.write(f"[Watch Trailer]({trailer_url})")
                if st.session_state.get(f"show_rating_{movie['id']}", False):
                    rating = st.slider(f"Rate {movie['title']} (1-5)", 1, 5, key=f"rating_genre_{movie['id']}")
                    if st.button("Submit Rating", key=f"submit_rating_genre_{movie['id']}"):
                        if st.session_state.current_user:
                            save_user_activity(st.session_state.current_user, "rated", movie['title'], movie['id'], rating)
                            st.success(f"Rated {movie['title']} with {rating} stars!")
                            st.session_state[f"show_rating_{movie['id']}"] = False
                        else:
                            st.warning("Please sign in to rate movies.")

    if search_query and not movies.empty:
        st.session_state.show_recommendations = False
        st.session_state.selected_genre = None
        filtered_movies = movies[movies['title'].str.contains(search_query, case=False, na=False)]
        if not filtered_movies.empty:
            cols = st.columns(3)
            for idx, movie in enumerate(filtered_movies.head(3).itertuples()):
                with cols[idx % 3]:
                    trailer_url = fetch_trailer(movie.id)
                    poster = fetch_poster(movie.id)
                    rating = movie.vote_average if hasattr(movie, 'vote_average') and pd.notna(movie.vote_average) else fetch_movie_details(movie.id)['rating']
                    description = movie.overview if hasattr(movie, 'overview') and pd.notna(movie.overview) else fetch_movie_details(movie.id)['description']
                    st.markdown(f"""
                        <div class="movie-card">
                            <img src="{poster}" style="width: 100%; border-radius: 10px;">
                            <h3>{movie.title}</h3>
                            <p>⭐ {rating:.1f}</p>
                            <p>{description[:100]}...</p>
                        </div>
                    """, unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Watch Now", key=f"watch_search_{movie.id}"):
                            if st.session_state.current_user:
                                save_user_activity(st.session_state.current_user, "watched", movie.title, movie.id)
                                st.session_state[f"rating_movie_{movie.id}"] = movie.title
                                st.session_state[f"show_rating_{movie.id}"] = True
                                st.session_state[f"rating_movie_id_{movie.id}"] = movie.id
                            else:
                                st.warning("Please sign in to watch movies.")
                    with col2:
                        if st.button("Add to Watchlist", key=f"watchlist_search_{movie.id}"):
                            if st.session_state.current_user:
                                if not any(item["title"] == movie.title for item in st.session_state.watchlist):
                                    st.session_state.watchlist.append({"title": movie.title, "movie_id": movie.id})
                                    save_watchlist_to_csv(st.session_state.current_user, movie.title, movie.id)
                                    save_user_activity(st.session_state.current_user, "added_to_watchlist", movie.title, movie.id)
                                    st.success(f"Added {movie.title} to watchlist!")
                            else:
                                st.warning("Please sign in to add movies to your watchlist.")
                    with col3:
                        if trailer_url:
                            if st.button("Watch Trailer", key=f"trailer_search_{movie.id}"):
                                st.write(f"[Watch Trailer]({trailer_url})")
                    if st.session_state.get(f"show_rating_{movie.id}", False):
                        rating = st.slider(f"Rate {movie.title} (1-5)", 1, 5, key=f"rating_search_{movie.id}")
                        if st.button("Submit Rating", key=f"submit_rating_search_{movie.id}"):
                            if st.session_state.current_user:
                                save_user_activity(st.session_state.current_user, "rated", movie.title, movie.id, rating)
                                st.success(f"Rated {movie.title} with {rating} stars!")
                                st.session_state[f"show_rating_{movie.id}"] = False
                            else:
                                st.warning("Please sign in to rate movies.")

    if st.session_state.show_recommendations:
        recommended_names = st.session_state.recommended_names
        recommended_posters = st.session_state.recommended_posters
        recommendation_type = st.session_state.recommendation_type
        st.subheader(f"{recommendation_type.capitalize()}-Based Recommendations")
        cols = st.columns(3)
        for idx, (name, poster) in enumerate(zip(recommended_names, recommended_posters)):
            with cols[idx % 3]:
                movie_id = movies[movies['title'] == name]['id'].iloc[0] if not movies.empty and name in movies['title'].values else None
                trailer_url = fetch_trailer(movie_id) if movie_id else None
                rating = movies[movies['title'] == name]['vote_average'].iloc[0] if not movies.empty and name in movies['title'].values and 'vote_average' in movies and pd.notna(movies[movies['title'] == name]['vote_average'].iloc[0]) else fetch_movie_details(movie_id)['rating'] if movie_id else 0.0
                description = movies[movies['title'] == name]['overview'].iloc[0] if not movies.empty and name in movies['title'].values and 'overview' in movies and pd.notna(movies[movies['title'] == name]['overview'].iloc[0]) else fetch_movie_details(movie_id)['description'] if movie_id else "No description available"
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{poster}" style="width: 100%; border-radius: 10px;">
                        <h3>{name}</h3>
                        <p>⭐ {rating:.1f}</p>
                        <p>{description[:100]}...</p>
                    </div>
                """, unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Watch Now", key=f"watch_rec_{idx}_{movie_id}"):
                        if st.session_state.current_user:
                            if movie_id:
                                save_user_activity(st.session_state.current_user, "watched", name, movie_id)
                                st.session_state[f"rating_movie_{movie_id}"] = name
                                st.session_state[f"show_rating_{movie_id}"] = True
                                st.session_state[f"rating_movie_id_{movie_id}"] = movie_id
                            else:
                                st.error("Cannot watch this movie: Movie ID not found.")
                        else:
                            st.warning("Please sign in to watch movies.")
                with col2:
                    if st.button("Add to Watchlist", key=f"watchlist_rec_{idx}_{movie_id}"):
                        if st.session_state.current_user:
                            if movie_id and not any(item["title"] == name for item in st.session_state.watchlist):
                                st.session_state.watchlist.append({"title": name, "movie_id": movie_id})
                                save_watchlist_to_csv(st.session_state.current_user, name, movie_id)
                                save_user_activity(st.session_state.current_user, "added_to_watchlist", name, movie_id)
                                st.success(f"Added {name} to watchlist!")
                            elif not movie_id:
                                st.error("Cannot add to watchlist: Movie ID not found.")
                        else:
                            st.warning("Please sign in to add movies to your watchlist.")
                with col3:
                    if trailer_url:
                        if st.button("Watch Trailer", key=f"trailer_rec_{idx}_{movie_id}"):
                            st.write(f"[Watch Trailer]({trailer_url})")
                if st.session_state.get(f"show_rating_{movie_id}", False):
                    rating = st.slider(f"Rate {name} (1-5)", 1, 5, key=f"rating_rec_{movie_id}")
                    if st.button("Submit Rating", key=f"submit_rating_rec_{movie_id}"):
                        if st.session_state.current_user:
                            if movie_id:
                                save_user_activity(st.session_state.current_user, "rated", name, movie_id, rating)
                                st.success(f"Rated {name} with {rating} stars!")
                                st.session_state[f"show_rating_{movie_id}"] = False
                            else:
                                st.error("Cannot rate this movie: Movie ID not found.")
                        else:
                            st.warning("Please sign in to rate movies.")
    else:
        if not movies.empty and not st.session_state.selected_genre and not search_query:
            cols = st.columns(3)
            for idx, movie in enumerate(movies.head(3).itertuples()):
                with cols[idx % 3]:
                    trailer_url = fetch_trailer(movie.id)
                    poster = fetch_poster(movie.id)
                    rating = movie.vote_average if hasattr(movie, 'vote_average') and pd.notna(movie.vote_average) else fetch_movie_details(movie.id)['rating']
                    description = movie.overview if hasattr(movie, 'overview') and pd.notna(movie.overview) else fetch_movie_details(movie.id)['description']
                    st.markdown(f"""
                        <div class="movie-card">
                            <img src="{poster}" style="width: 100%; border-radius: 10px;">
                            <h3>{movie.title}</h3>
                            <p>⭐ {rating:.1f}</p>
                            <p>{description[:100]}...</p>
                        </div>
                    """, unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Watch Now", key=f"watch_{movie.id}"):
                            if st.session_state.current_user:
                                save_user_activity(st.session_state.current_user, "watched", movie.title, movie.id)
                                st.session_state[f"rating_movie_{movie.id}"] = movie.title
                                st.session_state[f"show_rating_{movie.id}"] = True
                                st.session_state[f"rating_movie_id_{movie.id}"] = movie.id
                            else:
                                st.warning("Please sign in to watch movies.")
                    with col2:
                        if st.button("Add to Watchlist", key=f"watchlist_{movie.id}"):
                            if st.session_state.current_user:
                                if not any(item["title"] == movie.title for item in st.session_state.watchlist):
                                    st.session_state.watchlist.append({"title": movie.title, "movie_id": movie.id})
                                    save_watchlist_to_csv(st.session_state.current_user, movie.title, movie.id)
                                    save_user_activity(st.session_state.current_user, "added_to_watchlist", movie.title, movie.id)
                                    st.success(f"Added {movie.title} to watchlist!")
                            else:
                                st.warning("Please sign in to add movies to your watchlist.")
                    with col3:
                        if trailer_url:
                            if st.button("Watch Trailer", key=f"trailer_{movie.id}"):
                                st.write(f"[Watch Trailer]({trailer_url})")
                    if st.session_state.get(f"show_rating_{movie.id}", False):
                        rating = st.slider(f"Rate {movie.title} (1-5)", 1, 5, key=f"rating_default_{movie.id}")
                        if st.button("Submit Rating", key=f"submit_rating_default_{movie.id}"):
                            if st.session_state.current_user:
                                save_user_activity(st.session_state.current_user, "rated", movie.title, movie.id, rating)
                                st.success(f"Rated {movie.title} with {rating} stars!")
                                st.session_state[f"show_rating_{movie.id}"] = False
                            else:
                                st.warning("Please sign in to rate movies.")

elif st.session_state.page == "mood":
    st.markdown("<h2>Mood-Based Recommendations</h2>", unsafe_allow_html=True)
    st.markdown("<p>Answer a few questions to get movie recommendations tailored to your mood and preferences. All fields are optional.</p>", unsafe_allow_html=True)
    
    with st.form("mood_form"):
        mood = st.selectbox("What’s your current mood?", ["", "Happy", "Sad", "Stressed", "Excited", "Relaxed", "Bored", "Angry"], index=0)
        motivation = st.selectbox("Are you looking for something motivational or uplifting?", ["", "Yes", "No", "Neutral"], index=0)
        watching_with = st.selectbox("Who are you watching with?", ["", "Alone", "Friends", "Family", "Partner", "Kids"], index=0)
        occasion = st.selectbox("Is this for a special occasion?", ["", "Date Night", "Casual", "Party", "Family Night", "None"], index=0)
        time = st.selectbox("How much time do you have?", ["", "Less than 1 hour", "1-2 hours", "2+ hours"], index=0)
        genre = st.selectbox("What genre are you in the mood for?", [""] + list(fetch_genres().values()), index=0)
        tone = st.selectbox("What kind of tone do you prefer?", ["", "Light-hearted", "Serious", "Emotional", "Fun", "Epic", "Thought-provoking"], index=0)
        romantic = st.selectbox("Are you looking for something romantic?", ["", "Yes", "No", "Maybe"], index=0)
        pace = st.selectbox("Do you want something fast-paced or slow-paced?", ["", "Fast-paced", "Slow-paced", "Balanced"], index=0)
        release = st.selectbox("Do you prefer newer releases or classics?", ["", "New (post-2010)", "Classics (pre-2010)", "No preference"], index=0)
        mature = st.selectbox("Are you okay with intense or mature themes?", ["", "Yes", "No", "Neutral"], index=0)
        
        submit_button = st.form_submit_button("Get Recommendations")
        
        if submit_button:
            answers = {
                "mood": mood if mood else None,
                "motivation": motivation if motivation else None,
                "watching_with": watching_with if watching_with else None,
                "occasion": occasion if occasion else None,
                "time": time if time else None,
                "genre": genre if genre else None,
                "tone": tone if tone else None,
                "romantic": romantic if romantic else None,
                "pace": pace if pace else None,
                "release": release if release else None,
                "mature": mature if mature else None
            }
            st.session_state.mood_answers = answers
            st.session_state.mood_recommendations = recommend_mood_based(answers, fetch_genres())
            if st.session_state.mood_recommendations:
                st.success("Recommendations generated based on your mood!")
            else:
                st.warning("No movies found for your preferences. Showing popular movies instead.")

    if st.session_state.mood_recommendations:
        st.subheader("Movies for Your Mood")
        cols = st.columns(3)
        for idx, movie in enumerate(st.session_state.mood_recommendations):
            with cols[idx % 3]:
                trailer_url = fetch_trailer(movie['id'])
                st.markdown(f"""
                    <div class="movie-card">
                        <img src="{movie['poster']}" style="width: 100%; border-radius: 10px;">
                        <h3>{movie['title']}</h3>
                        <p>⭐ {movie['rating']:.1f}</p>
                        <p>{movie['description'][:100]}...</p>
                    </div>
                """, unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Watch Now", key=f"watch_mood_{movie['id']}"):
                        if st.session_state.current_user:
                            save_user_activity(st.session_state.current_user, "watched", movie['title'], movie['id'])
                            st.session_state[f"rating_movie_{movie['id']}"] = movie['title']
                            st.session_state[f"show_rating_{movie['id']}"] = True
                            st.session_state[f"rating_movie_id_{movie['id']}"] = movie['id']
                        else:
                            st.warning("Please sign in to watch movies.")
                with col2:
                    if st.button("Add to Watchlist", key=f"watchlist_mood_{movie['id']}"):
                        if st.session_state.current_user:
                            if not any(item["title"] == movie['title'] for item in st.session_state.watchlist):
                                st.session_state.watchlist.append({"title": movie['title'], "movie_id": movie['id']})
                                save_watchlist_to_csv(st.session_state.current_user, movie['title'], movie['id'])
                                save_user_activity(st.session_state.current_user, "added_to_watchlist", movie['title'], movie['id'])
                                st.success(f"Added {movie['title']} to watchlist!")
                        else:
                            st.warning("Please sign in to add movies to your watchlist.")
                with col3:
                    if trailer_url:
                        if st.button("Watch Trailer", key=f"trailer_mood_{movie['id']}"):
                            st.write(f"[Watch Trailer]({trailer_url})")
                if st.session_state.get(f"show_rating_{movie['id']}", False):
                    rating = st.slider(f"Rate {movie['title']} (1-5)", 1, 5, key=f"rating_mood_{movie['id']}")
                    if st.button("Submit Rating", key=f"submit_rating_mood_{movie['id']}"):
                        if st.session_state.current_user:
                            save_user_activity(st.session_state.current_user, "rated", movie['title'], movie['id'], rating)
                            st.success(f"Rated {movie['title']} with {rating} stars!")
                            st.session_state[f"show_rating_{movie['id']}"] = False
                        else:
                            st.warning("Please sign in to rate movies.")
    else:
        st.info("No mood-based recommendations available. Please submit your preferences or adjust filters.")

elif st.session_state.page == "watchlist":
    st.markdown("<h2>Watchlist</h2>", unsafe_allow_html=True)
    if st.session_state.current_user:
        if st.session_state.watchlist:
            st.markdown("<div class='watchlist-container'>", unsafe_allow_html=True)
            cols = st.columns(3)
            for idx, item in enumerate(st.session_state.watchlist):
                with cols[idx % 3]:
                    movie = item["title"]
                    movie_id = item["movie_id"]
                    poster = fetch_poster(movie_id) if movie_id else "https://via.placeholder.com/200x300?text=No+Poster"
                    trailer_url = fetch_trailer(movie_id) if movie_id else None
                    rating = movies[movies['id'] == movie_id]['vote_average'].iloc[0] if not movies.empty and movie_id in movies['id'].values and 'vote_average' in movies and pd.notna(movies[movies['id'] == movie_id]['vote_average'].iloc[0]) else fetch_movie_details(movie_id)['rating']
                    description = movies[movies['id'] == movie_id]['overview'].iloc[0] if not movies.empty and movie_id in movies['id'].values and 'overview' in movies and pd.notna(movies[movies['id'] == movie_id]['overview'].iloc[0]) else fetch_movie_details(movie_id)['description']
                    st.markdown(f"""
                        <div class="movie-card">
                            <img src="{poster}" style="width: 100%; border-radius: 10px;">
                            <h3>{movie}</h3>
                            <p>⭐ {rating:.1f}</p>
                            <p>{description[:100]}...</p>
                        </div>
                    """, unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Watch Now", key=f"watch_wl_{movie_id}_{idx}"):
                            save_user_activity(st.session_state.current_user, "watched", movie, movie_id)
                            st.session_state[f"rating_movie_{movie_id}"] = movie
                            st.session_state[f"show_rating_{movie_id}"] = True
                            st.session_state[f"rating_movie_id_{movie_id}"] = movie_id
                    with col2:
                        if st.button("Remove", key=f"remove_wl_{movie_id}_{idx}"):
                            st.session_state.watchlist = [item for item in st.session_state.watchlist if item["movie_id"] != movie_id]
                            filename = f"watchlist_{st.session_state.current_user}.csv"
                            if st.session_state.watchlist:
                                pd.DataFrame(st.session_state.watchlist).to_csv(filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
                            else:
                                if os.path.exists(filename):
                                    os.remove(filename)
                            st.success(f"Removed {movie} from watchlist!")
                    with col3:
                        if trailer_url:
                            if st.button("Trailer", key=f"trailer_wl_{movie_id}_{idx}"):
                                st.write(f"[Watch Trailer]({trailer_url})")
                    if st.session_state.get(f"show_rating_{movie_id}", False):
                        rating = st.slider(f"Rate {movie} (1-5)", 1, 5, key=f"rating_wl_{movie_id}_{idx}")
                        if st.button("Submit Rating", key=f"submit_rating_wl_{movie_id}_{ confronted_idx}"):
                            save_user_activity(st.session_state.current_user, "rated", movie, movie_id, rating)
                            st.success(f"Rated {movie} with {rating} stars!")
                            st.session_state[f"show_rating_{movie_id}"] = False
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Your watchlist is empty.")
    else:
        st.warning("Please sign in to view your watchlist.")

elif st.session_state.page == "history":
    st.markdown("<h2>Your Activity History</h2>", unsafe_allow_html=True)
    if st.session_state.current_user:
        if os.path.exists("user_activity.csv"):
            try:
                activity_df = pd.read_csv("user_activity.csv")
                user_activity = activity_df[activity_df['user_id'] == st.session_state.current_user]
                user_activity = user_activity.sort_values(by="timestamp", ascending=False)
                if user_activity.empty:
                    st.info("No activity history found.")
                else:
                    st.markdown("<div class='history-container'>", unsafe_allow_html=True)
                    cols = st.columns(3)
                    for idx, row in user_activity.iterrows():
                        with cols[idx % 3]:
                            action = row['action']
                            title = row['title']
                            movie_id = row['movie_id']
                            rating = row['rating'] if pd.notna(row['rating']) else None
                            timestamp = row['timestamp']
                            poster = fetch_poster(movie_id) if movie_id else "https://via.placeholder.com/200x300?text=No+Poster"
                            description = fetch_movie_details(movie_id)['description']
                            if action == "watched":
                                action_text = f"Watched on {timestamp}"
                            elif action == "rated":
                                action_text = f"Rated {rating}/5 on {timestamp}"
                            elif action == "added_to_watchlist":
                                action_text = f"Added to watchlist on {timestamp}"
                            else:
                                action_text = f"Unknown action on {timestamp}"
                            st.markdown(f"""
                                <div class="movie-card">
                                    <img src="{poster}" style="width: 100%; border-radius: 10px;">
                                    <h3>{title}</h3>
                                    <p>{action_text}</p>
                                    <p>{description[:100]}...</p>
                                </div>
                            """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Error reading activity history: {e}")
        else:
            st.info("No activity history yet.")
    else:
        st.warning("Please sign in to view your activity history.")
elif st.session_state.page == "signin":
    st.markdown("<h2>Authentication</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sign In")
        username = st.text_input("Username", key="signin_username")
        password = st.text_input("Password", type="password", key="signin_password")
        if st.button("Sign In", key="signin_button"):
            if username in st.session_state.users and bcrypt.checkpw(password.encode(), st.session_state.users[username]["password"].encode()):
                st.session_state.current_user = st.session_state.users[username]["user_id"]
                st.session_state.current_username = username
                st.session_state.watchlist = load_watchlist_from_csv(st.session_state.current_user)
                st.success(f"Welcome back, {username}!")
                st.session_state.page = "home"
            else:
                st.error("Invalid username or password.")
    with col2:
        st.subheader("Sign Up")
        new_username = st.text_input("New Username", key="signup_username")
        new_password = st.text_input("New Password", type="password", key="signup_password")
        if st.button("Sign Up", key="signup_button"):
            if new_username in st.session_state.users:
                st.error("Username already exists. Please choose another.")
            elif new_username and new_password:
                user_id = len(st.session_state.users) + 1
                hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                st.session_state.users[new_username] = {"password": hashed_password, "user_id": user_id}
                save_user_to_csv(new_username, new_password, user_id)
                st.session_state.current_user = user_id
                st.session_state.current_username = new_username
                st.session_state.watchlist = []
                st.session_state.mood_answers = {}
                st.session_state.mood_recommendations = []
                st.success(f"Account created successfully, {new_username}! Welcome to MovieMind!")
                st.session_state.page = "home"
            else:
                st.error("Please provide both a username and password.")