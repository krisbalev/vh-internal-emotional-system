from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import requests
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from transformers import pipeline
import numpy as np
from typing import Dict, List
import threading
import time
from scipy.optimize import minimize  # For weight optimization
import os
from openai import OpenAI
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# Global Parameters and Initial State
# ------------------------------------------------------------------------------

BASE_ALPHA = 0.2       # Base sensitivity factor (will be modulated dynamically).
DECAY_RATE = 0.02      # Decay rate: fraction of the difference toward baseline per second.
mood_state = [0.0, 0.0, 0.0]  # Global mood state in PAD space (initially neutral).
last_biased_emotion = None   # Stores the last computed personality-biased emotion vector.

# ------------------------------------------------------------------------------
# Combined Emotion Mapping: GoEmotions (27 labels) with Additional Hoffmann Coordinates
# ------------------------------------------------------------------------------

advanced_emotion_to_PAD = {
    "admiration":      (0.49, -0.19, 0.05),    # Hoffmann
    "amusement":       (0.45, 0.25, 0.0),      # GoEmotions
    "anger":           (-0.62, 0.59, 0.23),    # Hoffmann
    "annoyance":       (-0.4, 0.4, 0.1),       # GoEmotions
    "approval":        (0.3, 0.1, 0.0),        # GoEmotions
    "caring":          (0.4, 0.1, 0.2),        # GoEmotions
    "confusion":       (0.0, 0.0, -0.1),       # GoEmotions
    "curiosity":       (0.2, 0.3, 0.0),        # GoEmotions
    "desire":          (0.4, 0.5, 0.2),        # GoEmotions
    "disappointment":  (-0.64, -0.17, -0.41),  # Hoffmann
    "disapproval":     (-0.3, 0.2, 0.0),       # GoEmotions
    "disgust":         (-0.4, 0.2, 0.1),       # GoEmotions
    "embarrassment":   (-0.2, -0.1, -0.1),     # GoEmotions
    "excitement":      (0.6, 0.7, 0.1),        # GoEmotions
    "fear":            (-0.74, 0.47, -0.62),    # Hoffmann
    "gratitude":       (0.69, -0.09, 0.05),     # Hoffmann
    "grief":           (-0.5, -0.4, -0.2),      # GoEmotions
    "joy":             (0.82, 0.43, 0.55),      # Hoffmann
    "love":            (0.80, 0.14, 0.30),      # Hoffmann
    "nervousness":     (-0.2, 0.3, -0.2),      # GoEmotions
    "optimism":        (0.4, 0.3, 0.0),        # GoEmotions
    "pride":           (0.72, 0.20, 0.57),      # Hoffmann
    "realization":     (0.1, 0.1, 0.0),        # GoEmotions
    "relief":          (0.73, -0.24, 0.06),     # Hoffmann
    "remorse":         (-0.42, -0.01, -0.35),   # Hoffmann
    "sadness":         (-0.4, -0.2, -0.3),      # GoEmotions
    "surprise":        (0.0, 0.6, 0.0),         # GoEmotions
    # Additional Hoffmann et al. (2012) emotions:
    "gloating":        (0.08, 0.11, 0.44),      # Hoffmann
    "gratification":   (0.39, -0.18, 0.41),     # Hoffmann
    "happy for":       (0.75, 0.17, 0.37),      # Hoffmann
    "hope":            (0.22, 0.28, -0.23),     # Hoffmann
    "satisfaction":    (0.65, -0.42, 0.35),     # Hoffmann
    "distress":        (-0.75, -0.31, -0.47),   # Hoffmann
    "fears confirmed": (-0.74, 0.42, -0.52),    # Hoffmann
    "hate":            (-0.52, 0.0, 0.28),      # Hoffmann
    "pity":            (-0.27, -0.24, 0.24),    # Hoffmann
    "reproach":        (-0.41, 0.47, 0.50),     # Hoffmann
    "resentment":      (-0.52, 0.0, 0.03),      # Hoffmann
    "shame":           (-0.66, 0.05, -0.63)     # Hoffmann
}

# ------------------------------------------------------------------------------
# Advanced Emotion Descriptions
# ------------------------------------------------------------------------------

emotion_descriptions = {
    "admiration":      "A feeling of respect and warm approval.",
    "amusement":       "Finding something funny or entertaining.",
    "anger":           "A strong feeling of displeasure or hostility.",
    "annoyance":       "A slight feeling of irritation.",
    "approval":        "A positive endorsement or acceptance.",
    "caring":          "A sense of concern and empathy.",
    "confusion":       "A state of being bewildered or unclear.",
    "curiosity":       "A strong desire to learn or know more.",
    "desire":          "A strong feeling of wanting something.",
    "disappointment":  "A feeling of sadness when expectations are not met.",
    "disapproval":     "A feeling of rejection or disfavor.",
    "disgust":         "A strong feeling of aversion or repulsion.",
    "embarrassment":   "A feeling of self-consciousness or awkwardness.",
    "excitement":      "A state of high energy and enthusiasm.",
    "fear":            "An unpleasant emotion caused by the threat of harm or danger.",
    "gratitude":       "A feeling of thankfulness and appreciation.",
    "grief":           "Deep sorrow, especially following a loss.",
    "joy":             "A state of great pleasure and happiness.",
    "love":            "An intense feeling of deep affection.",
    "nervousness":     "A state of anxiousness or worry.",
    "optimism":        "Hopefulness and confidence about the future.",
    "pride":           "A sense of satisfaction from one's achievements or qualities.",
    "realization":     "The moment of understanding something clearly.",
    "relief":          "A feeling of reassurance following the end of anxiety or distress.",
    "remorse":         "Deep regret or guilt for a wrong committed.",
    "sadness":         "A feeling of sorrow or unhappiness.",
    "surprise":        "A feeling of shock or astonishment.",
    # Additional Hoffmann et al. (2012) descriptions:
    "gloating":        "A feeling of malicious self-satisfaction, often at someone else's misfortune.",
    "gratification":   "A sense of pleasure or satisfaction from achieving a desire or goal.",
    "happy for":       "Feeling joyful on behalf of someone else's good fortune.",
    "hope":            "An optimistic state based on the expectation of positive outcomes.",
    "satisfaction":    "A feeling of contentment or fulfillment when expectations are met.",
    "distress":        "An overwhelming sense of anxiety, sorrow, or pain.",
    "fears confirmed": "An emotion experienced when one's anxieties or fears are validated by events.",
    "hate":            "An intense feeling of dislike or hostility.",
    "pity":            "A feeling of sorrow or compassion caused by the suffering of others.",
    "reproach":        "A feeling of disapproval or disappointment towards someone’s actions.",
    "resentment":      "A bitter indignation resulting from unfair treatment.",
    "shame":           "A painful feeling of humiliation or distress due to guilt or inadequacy."
}

# ------------------------------------------------------------------------------
# Initialize the Emotion Classifier
# ------------------------------------------------------------------------------

classifier = pipeline(
    task="text-classification", 
    model="SamLowe/roberta-base-go_emotions",   # Updated model
    top_k=None   # Return all emotion scores
)

# ------------------------------------------------------------------------------
# Weight Calculation and Optimization (from the PCMD paper)
# ------------------------------------------------------------------------------
def compute_emotion_weights(personality, emotion_mapping, lambda_val=0.01):
    """
    Computes optimal emotion weights by solving:
         min_{w >= 0} ||D * w - P||^2 + lambda * ||w||^2,
    where D is a 3 x S matrix built from the emotion_mapping (PAD vectors)
    and P is the personality vector in PAD space.
    
    Args:
        personality: A list or array of 3 values (PAD vector) representing personality.
        emotion_mapping: A dictionary mapping emotion names to their PAD vectors.
        lambda_val: Regularization parameter.
        
    Returns:
        optimal_weights: A numpy array of optimal weights (one per emotion).
        emotion_names: The list of emotion names corresponding to each weight.
    """
    emotion_names = list(emotion_mapping.keys())
    S = len(emotion_names)
    # Build the matrix D of shape (3, S); each column is an emotion's PAD vector.
    D = np.array([emotion_mapping[emo] for emo in emotion_names]).T  # shape (3, S)
    
    P = np.array(personality)  # personality PAD vector (shape: (3,))
    
    # Define the objective function: ||D * w - P||^2 + lambda * ||w||^2.
    def objective(w):
        return np.linalg.norm(D.dot(w) - P)**2 + lambda_val * np.linalg.norm(w)**2
    
    # Initial guess: start with all ones.
    w0 = np.ones(S)
    
    # Define bounds: all weights must be nonnegative.
    bounds = [(0, None)] * S
    
    # Solve the optimization problem.
    res = minimize(objective, w0, bounds=bounds)
    
    if res.success:
        optimal_weights = res.x
    else:
        raise ValueError("Optimization did not converge.")
    
    return optimal_weights, emotion_names

# ------------------------------------------------------------------------------
# Advanced Emotion Detection using Optimal Weights
# ------------------------------------------------------------------------------
def detect_emotion_weighted(text: str, optimal_weights: dict):
    """
    Detects emotions using the transformer-based classifier and applies optimal weights.
    It aggregates predictions weighted by (score * optimal_weight) to compute a composite PAD vector.
    
    Returns:
        composite_PAD: The weighted average PAD vector.
        predictions: The raw predictions from the classifier.
    """
    predictions = classifier(text)[0]
    composite_PAD = [0.0, 0.0, 0.0]
    total_weighted_score = 0.0
    for pred in predictions:
        label = pred["label"].lower()  # Normalize label to match mapping keys.
        score = pred["score"]
        weight = optimal_weights.get(label, 1.0)  # Default to 1.0 if label not in optimal_weights.
        total_weighted_score += score * weight
        pad = advanced_emotion_to_PAD.get(label, (0.0, 0.0, 0.0))
        composite_PAD[0] += score * weight * pad[0]
        composite_PAD[1] += score * weight * pad[1]
        composite_PAD[2] += score * weight * pad[2]
    if total_weighted_score > 0:
        composite_PAD = [x / total_weighted_score for x in composite_PAD]
    return composite_PAD, predictions

# ------------------------------------------------------------------------------
# Update Mood Functions
# ------------------------------------------------------------------------------
def update_mood(current_mood, biased_emotion, alpha):
    """
    Updates the global mood state using exponential smoothing:
        new_mood = (1 - alpha) * current_mood + alpha * biased_emotion
    This ensures that if the biased emotion is close to the current mood, the update is minimal,
    while large differences result in a more significant update.
    
    Returns:
        New mood state (each component clamped between -1 and 1).
    """
    new_mood = [(1 - alpha) * c + alpha * b for c, b in zip(current_mood, biased_emotion)]
    return [max(-1, min(1, val)) for val in new_mood]

def mood_to_description(mood):
    """
    Converts the numeric mood state (PAD vector) into a natural language description.
    """
    p, a, d = mood
    if p > 0.3:
        valence = "happy"
    elif p < -0.3:
        valence = "sad"
    else:
        valence = "neutral"
    if a > 0.3:
        energy = "energetically"
    elif a < -0.3:
        energy = "calmly"
    else:
        energy = "moderately"
    if d > 0.3:
        control = "and in control"
    elif d < -0.3:
        control = "and overwhelmed"
    else:
        control = "with balance"
    return f"{energy} {valence} {control}"

def get_top_dominant_emotions(current_pad, emotion_map, top_n=3):
    """
    Returns the top_n emotions (based on Euclidean distance) closest to the current PAD vector.
    """
    similarities = []
    for emotion, vec in emotion_map.items():
        dist = np.linalg.norm(np.array(current_pad) - np.array(vec))
        similarities.append((emotion, dist))
    similarities.sort(key=lambda x: x[1])
    return similarities[:top_n]

# ------------------------------------------------------------------------------
# Virtual Human Class with Personality and Mood Biasing
# ------------------------------------------------------------------------------
PADVector = List[float]
BigFive = Dict[str, float]

class VirtualHuman:
    def __init__(self, personality: BigFive, personality_bias: float = 0.4, mood_bias: float = 0.2) -> None:
        """
        personality_bias: Weight for personality influence.
        mood_bias: Weight for current mood influence.
        (The remaining weight, 1 - personality_bias - mood_bias, is for the raw user emotion.)
        """
        self.personality = personality
        self.personality_bias = personality_bias
        self.mood_bias = mood_bias
        # Compute personality baseline from Big Five mapping.
        self.personality_PAD = self.bigfive_to_PAD()
        # Compute optimal emotion weights based on personality and emotion mapping.
        optimal_weights, emotion_names = compute_emotion_weights(self.personality_PAD, advanced_emotion_to_PAD, lambda_val=0.01)
        self.optimal_weights = {name: weight for name, weight in zip(emotion_names, optimal_weights)}
    
    def bigfive_to_PAD(self) -> PADVector:
        """
        Maps the Big Five personality traits to a PAD vector using Mehrabian's regression formulas.
        Assumptions:
          - Emotional Stability is calculated as 1 - neuroticism.
          - Sophistication is taken as openness.
          - Big Five scores are assumed to be on a scale from 0 to 1.
        """
        extraversion = self.personality.get("extraversion", 0.5)
        agreeableness = self.personality.get("agreeableness", 0.5)
        neuroticism = self.personality.get("neuroticism", 0.5)
        conscientiousness = self.personality.get("conscientiousness", 0.5)
        # Convert Neuroticism to Emotional Stability:
        emotional_stability = 1 - neuroticism
        sophistication = self.personality.get("openness", 0.5)  # treating openness as sophistication

        # Apply Mehrabian's regression equations:
        P = 0.59 * agreeableness + 0.19 * emotional_stability + 0.21 * extraversion
        A = -0.57 * emotional_stability + 0.30 * agreeableness + 0.15 * sophistication
        D = 0.60 * extraversion - 0.32 * agreeableness + 0.25 * sophistication + 0.17 * conscientiousness

        # Optionally, clip values to [-1, 1] if needed.
        return [np.clip(P, -1, 1), np.clip(A, -1, 1), np.clip(D, -1, 1)]
    
    def compute_immediate_reaction(self, user_emotion: PADVector, current_mood: PADVector) -> PADVector:
        """
        Computes the immediate emotion reaction by blending:
          - The raw weighted emotion (from user input processed with optimal weights)
          - The current mood (as an appraisal bias)
        The blending weights are:
          (1 - mood_bias) * user_emotion + mood_bias * current_mood
        """
        user_vec = np.array(user_emotion)
        current_mood_vec = np.array(current_mood)
        final_vec = (1 - self.mood_bias) * user_vec + self.mood_bias * current_mood_vec
        return np.clip(final_vec, -1, 1).tolist()
    
    def generate_response(self, user_text: str, final_emotion: PADVector) -> str:
        """
        Converts the final PAD vector into a natural language description and uses the user input.
        """
        dominant_emotions_data = get_top_dominant_emotions(final_emotion, advanced_emotion_to_PAD, top_n=3)
        dominant_emotion_names = [emotion for emotion, _ in dominant_emotions_data]
        return generate_chatgpt_response(user_text, final_emotion, dominant_emotion_names, self.personality)

# ------------------------------------------------------------------------------
# ChatGPT API Integration
# ------------------------------------------------------------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def generate_chatgpt_response(user_text, current_mood, dominant_emotions, personality):
    """
    Generates a response from ChatGPT by providing both the user input and the virtual human's mood context.
    
    Args:
        user_text (str): The user's input.
        current_mood (list): The current PAD state, e.g., [pleasure, arousal, dominance].
        mood_description (str): A natural language description of the mood (e.g., "energetically happy and in control").
    
    Returns:
        response (str): The generated response from ChatGPT.
    """
    personality_description = bigfive_to_text(personality)
    system_prompt = (
        f"You are Viktor, a virtual agent with a distinct personality and an internal mood state. "
        f"Your personality is characterized as: {personality_description}. "
        f"Your current emotion is represented by the PAD vector {current_mood}. "
        f"Your dominant emotions are {', '.join(dominant_emotions)}. "
        "When responding, reflect both your personality and your current affective state in your tone and word choice."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling ChatGPT API: {e}"
    

def bigfive_to_text(bigfive: Dict[str, float]) -> str:
    extraversion = bigfive.get("extraversion", 0.5)
    neuroticism = bigfive.get("neuroticism", 0.5)
    openness = bigfive.get("openness", 0.5)
    agreeableness = bigfive.get("agreeableness", 0.5)
    conscientiousness = bigfive.get("conscientiousness", 0.5)
    
    description = []
    
    # Extraversion
    if extraversion > 0.7:
        description.append("highly extraverted")
    elif extraversion < 0.3:
        description.append("introverted")
    else:
        description.append("moderately extraverted")
    
    # Neuroticism (inverse for stability)
    if neuroticism > 0.7:
        description.append("prone to anxiety")
    elif neuroticism < 0.3:
        description.append("emotionally stable")
    else:
        description.append("moderately emotionally stable")
        
    # Openness
    if openness > 0.7:
        description.append("very open to new experiences")
    elif openness < 0.3:
        description.append("more conventional")
    else:
        description.append("moderately open")
    
    # Agreeableness
    if agreeableness > 0.7:
        description.append("highly cooperative and empathetic")
    elif agreeableness < 0.3:
        description.append("more competitive")
    else:
        description.append("fairly agreeable")
    
    # Conscientiousness
    if conscientiousness > 0.7:
        description.append("very conscientious")
    elif conscientiousness < 0.3:
        description.append("laid-back")
    else:
        description.append("moderately conscientious")
    
    return ", ".join(description)

# ------------------------------------------------------------------------------
# Dynamic Alpha Calculation
# ------------------------------------------------------------------------------
def compute_dynamic_alpha(current_mood, new_emotion, base_alpha=BASE_ALPHA):
    """
    Computes an update factor (alpha) based on the cosine similarity between the current mood
    and the new (personality-biased) emotion vector.
    """
    current = np.array(current_mood)
    new = np.array(new_emotion)
    norm_current = np.linalg.norm(current)
    norm_new = np.linalg.norm(new)
    if norm_current == 0 or norm_new == 0:
        return base_alpha
    cos_sim = np.dot(current, new) / (norm_current * norm_new)
    new_alpha = 0.3 * (cos_sim + 1) / 2 + 0.1
    return new_alpha

# ------------------------------------------------------------------------------
# Plotting and Visualization
# ------------------------------------------------------------------------------
def update_plot(ax, mood, biased_emotion):
    """
    Redraws the 3D plot showing:
      - All static emotion positions (with labels) from advanced_emotion_to_PAD.
      - The current global mood (red dot).
      - The personality-biased emotion vector (blue arrow) from the origin.
    """
    ax.cla()  # Clear previous plot.
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('Pleasure')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    ax.set_title("Current Mood & Personality-Biased Emotion in PAD Space")
    
    for emotion, coord in advanced_emotion_to_PAD.items():
        ax.scatter(coord[0], coord[1], coord[2], c='gray', marker='^', s=50)
        ax.text(coord[0], coord[1], coord[2], emotion, size=8, color='black')
    
    ax.scatter(mood[0], mood[1], mood[2], c='r', marker='o', s=100, label='Global Mood')
    if biased_emotion is not None:
        ax.quiver(0, 0, 0, biased_emotion[0], biased_emotion[1], biased_emotion[2],
                  color='b', arrow_length_ratio=0.1, label='Biased Emotion Vector')
    ax.legend()
    plt.draw()

# ------------------------------------------------------------------------------
# Timer Callback for Continuous Mood Decay
# ------------------------------------------------------------------------------
def decay_callback():
    """
    Applies exponential decay to the current mood, moving it gradually toward the personality baseline.
    """
    global mood_state
    personality_baseline = vh.bigfive_to_PAD()
    mood_state[:] = [current + DECAY_RATE * (baseline - current)
                     for current, baseline in zip(mood_state, personality_baseline)]
    update_plot(ax, mood_state, last_biased_emotion)

# ------------------------------------------------------------------------------
# Background Input Thread for User Interaction
# ------------------------------------------------------------------------------
def process_input():
    """
    Continuously reads user input from the console and processes the emotion,
    updating the mood state using the blended reaction.
    """
    global mood_state, last_biased_emotion
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        # Use the weighted emotion detection with optimal weights.
        composite_PAD, predictions = detect_emotion_weighted(user_input, vh.optimal_weights)
        print("\nDetected Emotion Predictions (Weighted):")
        top_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)[:3]
        for pred in top_preds:
            print(f"  {pred['label']} : {pred['score']:.2f}")
        print(f"Composite Emotion PAD vector (weighted): {composite_PAD}")
        
        # Compute the personality- and mood-biased reaction.
        biased_emotion = vh.compute_immediate_reaction(composite_PAD, mood_state)
        last_biased_emotion = biased_emotion
        print(f"Personality- & Mood-biased Emotion PAD vector: {biased_emotion}")
        
        # Compute dynamic update factor (alpha).
        dynamic_alpha = compute_dynamic_alpha(mood_state, biased_emotion, base_alpha=BASE_ALPHA)
        print(f"Dynamic update factor (alpha): {dynamic_alpha:.2f}")
        
        # Update the global mood using exponential smoothing.
        mood_state[:] = update_mood(mood_state, biased_emotion, dynamic_alpha)
        print(f"Updated Global Mood (PAD): {mood_state} -> {mood_to_description(mood_state)}")
        
        # Generate and print Viktor's response using ChatGPT API
        response = vh.generate_response(user_input, mood_state)
        print("Viktor AI Response:", response)
        
        # Display the top three dominant emotions.
        dominant_emotions = get_top_dominant_emotions(mood_state, advanced_emotion_to_PAD, top_n=3)
        print("Dominant Emotions:")
        for emotion, dist in dominant_emotions:
            explanation = emotion_descriptions.get(emotion, "No description available.")
            print(f"  {emotion.capitalize()} (distance: {dist:.2f}): {explanation}")
        
        update_plot(ax, mood_state, last_biased_emotion)

        socketio.emit("new_message", {"message": user_input, "response": response})

# ------------------------------------------------------------------------------
# Flask Application and Socket.IO Setup
# ------------------------------------------------------------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

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
    
    # Detect emotion from the user input using weighted detection.
    composite_PAD, predictions = detect_emotion_weighted(user_text, vh.optimal_weights)
    # Compute personality- and mood-biased reaction.
    biased_emotion = vh.compute_immediate_reaction(composite_PAD, mood_state)
    last_biased_emotion = biased_emotion
    
    # Compute dynamic update factor.
    dynamic_alpha = compute_dynamic_alpha(mood_state, biased_emotion, base_alpha=BASE_ALPHA)
    
    # Update the global mood.
    mood_state[:] = update_mood(mood_state, biased_emotion, dynamic_alpha)
    
    # Generate a response using ChatGPT integration.
    response = vh.generate_response(user_text, mood_state)
    
    # Emit the response to WebSocket clients.
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
        # Send TTS request to ElevenLabs
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        payload = {
            "text": text,
            "voice_settings": {
                "stability": 0.75,
                "similarity_boost": 0.9
            }
        }

        response = requests.post(
            f"{ELEVENLABS_API_URL}/{VOICE_ID}",
            json=payload,
            headers=headers
        )

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

# ------------------------------------------------------------------------------
# Main Setup and Event Loop
# ------------------------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a timer (every 1000 ms) to apply mood decay.
timer = fig.canvas.new_timer(interval=1000)
timer.add_callback(decay_callback)
timer.start()

# Define Viktor's personality using the Big Five traits.
personality: BigFive = {
    "extraversion": 0.3,
    "neuroticism": 0.4,
    "openness": 0.9,
    "agreeableness": 0.5,
    "conscientiousness": 0.9
}
# Create the VirtualHuman (Viktor) with personality_bias and mood_bias.
vh = VirtualHuman(personality=personality, personality_bias=0.4, mood_bias=0.2)
# Initialize the global mood_state with the personality baseline.
mood_state = vh.bigfive_to_PAD()

print("Viktor AI Emotion Reaction System. Type 'exit' to quit.")
update_plot(ax, mood_state, last_biased_emotion)

input_thread = threading.Thread(target=process_input)
input_thread.start()

if __name__ == "__main__":
    threading.Thread(target=lambda: socketio.run(app, host="0.0.0.0", port=5000)).start()
    plt.show(block=True)
