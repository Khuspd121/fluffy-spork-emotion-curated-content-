import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import requests
import random # NEW: Imported the random library

# --- 1. Initialize Flask Application ---
app = Flask(__name__)
CORS(app)

# --- ACTION REQUIRED: PASTE YOUR API KEYS HERE ---
TMDB_API_KEY = "95e4c527f23c8fc22bc14b78c958978a"
SPOTIFY_CLIENT_ID = "2eff6d9c29324b21a18afa063d43043f"
SPOTIFY_CLIENT_SECRET = "6462e282bfd84c61a1d6c78f1c266d79"
# ----------------------------------------------------

# --- 2. Load the Trained Emotion Detection Model ---
try:
    model = tf.keras.models.load_model('emotion_model.h5')
    print("✅ Emotion detection model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

EMOTION_MAPPING = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# --- 3. Recommendation Logic ---

# --- Spotify ---
def get_spotify_token():
    """Gets an access token from the Spotify API."""
    auth_url = 'https://accounts.spotify.com/api/token'
    try:
        auth_response = requests.post(auth_url, {
            'grant_type': 'client_credentials',
            'client_id': SPOTIFY_CLIENT_ID,
            'client_secret': SPOTIFY_CLIENT_SECRET,
        }, timeout=10)
        auth_response.raise_for_status()
        auth_response_data = auth_response.json()
        return auth_response_data.get('access_token')
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting Spotify token: {e}")
        return None


def get_music_recommendations(emotion, token):
    """Gets music recommendations from Spotify based on emotion."""
    search_url = "https://api.spotify.com/v1/search"

    emotion_to_genre = {
        "Happy": "pop,happy,summer",
        "Sad": "sad,acoustic,rainy day",
        "Angry": "rock,metal,angry",
        "Surprise": "electronic,dance,party",
        "Fear": "ambient,classical,instrumental",
        "Neutral": "chill,focus,instrumental",
        "Disgust": "punk,industrial"
    }

    query = emotion_to_genre.get(emotion, "chill")

    headers = {'Authorization': f'Bearer {token}'}
    # NEW: Added a random offset to get different results each time
    params = {'q': query, 'type': 'track', 'limit': 5, 'offset': random.randint(0, 50)}

    try:
        response = requests.get(search_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()

        tracks = response.json().get('tracks', {}).get('items', [])
        recommendations = []
        for track in tracks:
            recommendations.append({
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'url': track['external_urls']['spotify'],
                'image': track['album']['images'][0]['url'] if track['album']['images'] else ''
            })
        return recommendations
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching music recommendations from Spotify: {e}")
        return []

# --- TMDb (Movies) ---
def get_movie_recommendations(emotion):
    """Gets movie recommendations from TMDb based on emotion."""
    emotion_to_genre_id = {
        "Happy": 35, "Sad": 18, "Angry": 28, "Surprise": 53,
        "Fear": 27, "Neutral": 99, "Disgust": 27
    }

    genre_id = emotion_to_genre_id.get(emotion, 18)

    discover_url = f"https://api.themoviedb.org/3/discover/movie"
    # NEW: Added a random page number to get different results each time
    params = {
        'api_key': TMDB_API_KEY,
        'with_genres': genre_id,
        'sort_by': 'popularity.desc',
        'page': random.randint(1, 10) # Get results from a random page
    }

    try:
        response = requests.get(discover_url, params=params, timeout=10)
        response.raise_for_status()

        movies = response.json().get('results', [])[:5]
        recommendations = []
        for movie in movies:
            if movie.get('poster_path'):
                recommendations.append({
                    'title': movie['title'],
                    'overview': movie['overview'],
                    'poster_path': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                })
        return recommendations
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching movie recommendations from TMDb: {e}")
        return []

# --- 4. Preprocess the Image for the Model ---
def preprocess_image(image_data):
    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes))
        image_np = np.array(image)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return None

        (x, y, w, h) = sorted(faces, reverse=True, key=lambda f: f[2] * f[3])[0]
        face_roi = gray_image[y:y + h, x:x + w]

        resized_face = cv2.resize(face_roi, (48, 48))
        normalized_face = resized_face / 255.0
        final_face = np.expand_dims(np.expand_dims(normalized_face, -1), 0)

        return final_face
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

# --- 5. Define the Prediction API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded!'}), 500

    data = request.get_json()
    # NEW: Handle requests for both initial prediction and refresh
    emotion = data.get('emotion')

    # If no emotion is provided, perform prediction
    if not emotion:
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        processed_image = preprocess_image(data['image'])
        if processed_image is None:
            return jsonify({'emotion': 'No Face Detected', 'movies': [], 'music': []})

        prediction = model.predict(processed_image)
        predicted_emotion_index = int(np.argmax(prediction))
        emotion = EMOTION_MAPPING[predicted_emotion_index]

    print(f"Getting recommendations for emotion: {emotion}")

    # Fetch recommendations
    movie_recs = get_movie_recommendations(emotion)
    spotify_token = get_spotify_token()
    music_recs = get_music_recommendations(emotion, spotify_token) if spotify_token else []

    return jsonify({
        'emotion': emotion,
        'movies': movie_recs,
        'music': music_recs
    })

# --- 6. Run the Flask Application ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
