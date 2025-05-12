from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
import requests
import os
import threading
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from virtual_human import VirtualHuman
from mood import update_plot, decay_callback, mood_to_description, compute_dynamic_alpha, update_mood, get_top_dominant_emotions
from legacy.legacy_project.emotion_processing import detect_emotion_weighted
from config import BASE_ALPHA, advanced_emotion_to_PAD, emotion_descriptions

load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for mood handling
mood_state = None
last_biased_emotion = None

# Initialize matplotlib figure for visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define Johnny Bravo's personality using Big Five traits.
personality = {
    "extraversion": 0.3,
    "neuroticism": 0.4,
    "openness": 0.9,
    "agreeableness": 0.5,
    "conscientiousness": 0.9
}

# Create the VirtualHuman instance
vh = VirtualHuman(personality=personality, personality_bias=0.4, mood_bias=0.2)
mood_state = vh.bigfive_to_PAD()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global mood_state, last_biased_emotion
    data = request.get_json()
    user_text = data.get("message", "")
    if not user_text:
        return jsonify({"error": "No message provided"}), 400
    
    composite_PAD, predictions = detect_emotion_weighted(user_text, vh.optimal_weights)
    biased_emotion = vh.compute_immediate_reaction(composite_PAD, mood_state)
    last_biased_emotion = biased_emotion
    dynamic_alpha = compute_dynamic_alpha(mood_state, biased_emotion, base_alpha=BASE_ALPHA)
    mood_state[:] = update_mood(mood_state, biased_emotion, dynamic_alpha)
    
    response = vh.generate_response(user_text, mood_state)
    socketio.emit("new_message", {"message": user_text, "response": response})
    update_plot(ax, mood_state, last_biased_emotion)
    
    return jsonify({
        "response": response,
        "mood": mood_state,
        "mood_description": mood_to_description(mood_state)
    })

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_API_URL = os.getenv("ELEVENLABS_API_URL")
VOICE_ID = "iP95p4xoKVk53GoZ742B"

@app.route('/tts', methods=['POST'])
def tts():
    """Generate TTS audio using ElevenLabs."""
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        payload = {
            "text": text,
            "voice_settings": {"stability": 0.75, "similarity_boost": 0.9}
        }
        response = requests.post(f"{ELEVENLABS_API_URL}/{VOICE_ID}", json=payload, headers=headers)
        if response.status_code == 200:
            audio_path = "output.mp3"
            with open(audio_path, "wb") as audio_file:
                audio_file.write(response.content)
            return send_file(audio_path, mimetype="audio/mpeg")
        else:
            print("ElevenLabs response:", response.text)
            return jsonify({"error": "TTS generation failed", "details": response.text}), 500
    except Exception as e:
        print("Error in /tts endpoint:", str(e))
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

@socketio.on("connect")
def handle_connect():
    print("Client connected")

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")

def process_input():
    """
    Background thread to read console input and update emotion/mood accordingly.
    """
    global mood_state, last_biased_emotion
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        composite_PAD, predictions = detect_emotion_weighted(user_input, vh.optimal_weights)
        print("\nDetected Emotion Predictions (Weighted):")
        top_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)[:3]
        for pred in top_preds:
            print(f"  {pred['label']} : {pred['score']:.2f}")
        print(f"Composite Emotion PAD vector (weighted): {composite_PAD}")
        
        biased_emotion = vh.compute_immediate_reaction(composite_PAD, mood_state)
        last_biased_emotion = biased_emotion
        print(f"Personality- & Mood-biased Emotion PAD vector: {biased_emotion}")
        
        dynamic_alpha = compute_dynamic_alpha(mood_state, biased_emotion, base_alpha=BASE_ALPHA)
        print(f"Dynamic update factor (alpha): {dynamic_alpha:.2f}")
        
        mood_state[:] = update_mood(mood_state, biased_emotion, dynamic_alpha)
        print(f"Updated Global Mood (PAD): {mood_state} -> {mood_to_description(mood_state)}")
        
        response = vh.generate_response(user_input, mood_state)
        print("Johnny Bravo AI Response:", response)
        
        dominant_emotions = get_top_dominant_emotions(mood_state, advanced_emotion_to_PAD, top_n=3)
        print("Dominant Emotions:")
        for emotion, dist in dominant_emotions:
            explanation = emotion_descriptions.get(emotion, "No description available.")
            print(f"  {emotion.capitalize()} (distance: {dist:.2f}): {explanation}")
        
        update_plot(ax, mood_state, last_biased_emotion)
        socketio.emit("new_message", {"message": user_input, "response": response})

if __name__ == "__main__":
    # Start the input thread as a daemon.
    input_thread = threading.Thread(target=process_input, daemon=True)
    input_thread.start()
    
    # Set up a matplotlib timer for periodic mood decay.
    timer = fig.canvas.new_timer(interval=1000)
    personality_baseline = vh.bigfive_to_PAD()
    timer.add_callback(lambda: decay_callback(mood_state, personality_baseline, ax, last_biased_emotion))
    timer.start()

    # Start the Flask-SocketIO server in a daemon thread.
    socketio_thread = threading.Thread(target=lambda: socketio.run(app, host="0.0.0.0", port=5000), daemon=True)
    socketio_thread.start()
    
    try:
        # Show the plot; this will block.
        plt.show(block=True)
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Shutting down...")
    finally:
        plt.close('all')
