import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from transformers import pipeline
import numpy as np
from typing import Dict, List
import threading
import time

# ------------------------------------------------------------------------------
# Global Parameters and Initial State
# ------------------------------------------------------------------------------

BASE_ALPHA = 0.2   # Base sensitivity factor used in updating mood.
DECAY = 0.98       # Decay factor applied every second to pull mood back to neutral.
mood_state = [0.0, 0.0, 0.0]  # Global mood state in PAD space (initially neutral).
last_biased_emotion = None   # Stores the last computed personality-biased emotion vector.

# ------------------------------------------------------------------------------
# Emotion Mapping and Descriptions
# ------------------------------------------------------------------------------

# Mapping from GoEmotions labels (27 emotions) to approximate PAD (Pleasure, Arousal, Dominance) vectors.
advanced_emotion_to_PAD = {
    "admiration":      (0.5, 0.3, -0.2),
    "amusement":       (0.45, 0.25, 0.0),
    "anger":           (-0.51, 0.59, 0.25),
    "annoyance":       (-0.4, 0.4, 0.1),
    "approval":        (0.3, 0.1, 0.0),
    "caring":          (0.4, 0.1, 0.2),
    "confusion":       (0.0, 0.0, -0.1),
    "curiosity":       (0.2, 0.3, 0.0),
    "desire":          (0.4, 0.5, 0.2),
    "disappointment":  (-0.3, 0.1, -0.4),
    "disapproval":     (-0.3, 0.2, 0.0),
    "disgust":         (-0.4, 0.2, 0.1),
    "embarrassment":   (-0.2, -0.1, -0.1),
    "excitement":      (0.6, 0.7, 0.1),
    "fear":            (-0.64, 0.60, -0.43),
    "gratitude":       (0.4, 0.2, -0.3),
    "grief":           (-0.5, -0.4, -0.2),
    "joy":             (0.4, 0.2, 0.1),
    "love":            (0.3, 0.1, 0.2),
    "nervousness":     (-0.2, 0.3, -0.2),
    "optimism":        (0.4, 0.3, 0.0),
    "pride":           (0.4, 0.3, 0.3),
    "realization":     (0.1, 0.1, 0.0),
    "relief":          (0.2, -0.3, 0.4),
    "remorse":         (-0.3, 0.1, -0.6),
    "sadness":         (-0.4, -0.2, -0.3),
    "surprise":        (0.0, 0.6, 0.0)
}

# Short descriptions for each emotion.
emotion_descriptions = {
    "admiration": "A feeling of respect and warm approval.",
    "amusement": "Finding something funny or entertaining.",
    "anger": "A strong feeling of annoyance or hostility.",
    "annoyance": "A slight feeling of irritation.",
    "approval": "A positive endorsement or acceptance.",
    "caring": "A sense of concern and empathy.",
    "confusion": "A state of being bewildered or unclear.",
    "curiosity": "A strong desire to learn or know more.",
    "desire": "A strong feeling of wanting something.",
    "disappointment": "A feeling of sadness from unmet expectations.",
    "disapproval": "A feeling of rejection or disfavor.",
    "disgust": "A strong feeling of aversion or repulsion.",
    "embarrassment": "A feeling of self-consciousness or awkwardness.",
    "excitement": "A state of high energy and enthusiasm.",
    "fear": "An unpleasant emotion due to a perceived threat.",
    "gratitude": "A feeling of thankfulness and appreciation.",
    "grief": "Deep sorrow, especially following a loss.",
    "joy": "A feeling of great pleasure or happiness.",
    "love": "An intense feeling of deep affection.",
    "nervousness": "A state of anxiousness or worry.",
    "optimism": "Hopefulness and confidence about the future.",
    "pride": "A feeling of satisfaction from achievements.",
    "realization": "The moment of understanding something clearly.",
    "relief": "A feeling of reassurance following stress.",
    "remorse": "Deep regret or guilt for something done.",
    "sadness": "A feeling of sorrow or unhappiness.",
    "surprise": "A feeling of shock or astonishment."
}

# ------------------------------------------------------------------------------
# Initialize the Emotion Classifier
# ------------------------------------------------------------------------------

# This pipeline uses a transformer model ("monologg/bert-base-cased-goemotions-original")
# to classify text into a range of emotions, returning labels and scores.
classifier = pipeline(
    "text-classification",
    model="monologg/bert-base-cased-goemotions-original",
    top_k=None,
    truncation=True
)

# ------------------------------------------------------------------------------
# Emotion Detection and Mood Update Functions
# ------------------------------------------------------------------------------

def detect_emotion_advanced(text: str):
    """
    Detects a broad range of emotions using the transformer-based classifier.
    It aggregates predictions weighted by their scores to compute a composite PAD vector.
    
    Returns:
        composite_PAD: The weighted average PAD vector.
        predictions: The raw predictions from the classifier.
    """
    predictions = classifier(text)[0]
    composite_PAD = [0.0, 0.0, 0.0]
    total_score = 0.0
    for pred in predictions:
        label = pred["label"].lower()  # Normalize label to match mapping keys.
        score = pred["score"]
        total_score += score
        pad = advanced_emotion_to_PAD.get(label, (0.0, 0.0, 0.0))
        # Multiply the PAD vector by the prediction score.
        composite_PAD[0] += score * pad[0]
        composite_PAD[1] += score * pad[1]
        composite_PAD[2] += score * pad[2]
    if total_score > 0:
        # Compute weighted average.
        composite_PAD = [x / total_score for x in composite_PAD]
    return composite_PAD, predictions

def update_mood(current_mood, emotion_PAD, alpha):
    """
    Updates the global mood state by moving it a fraction (alpha) toward the given emotion PAD vector.
    
    Args:
        current_mood: Current mood state vector.
        emotion_PAD: New emotion vector (after personality biasing).
        alpha: Update factor.
    
    Returns:
        New mood state (each component clamped between -1 and 1).
    """
    new_mood = [
        current_mood[0] + alpha * emotion_PAD[0],
        current_mood[1] + alpha * emotion_PAD[1],
        current_mood[2] + alpha * emotion_PAD[2]
    ]
    # Clamp each value between -1 and 1.
    return [max(-1, min(1, val)) for val in new_mood]

def mood_to_description(mood):
    """
    Converts the numeric mood state (PAD vector) into a natural language description.
    
    Args:
        mood: The mood state vector [pleasure, arousal, dominance].
    
    Returns:
        A descriptive string.
    """
    p, a, d = mood
    # Determine valence (pleasure)
    if p > 0.3:
        valence = "happy"
    elif p < -0.3:
        valence = "sad"
    else:
        valence = "neutral"
    # Determine energy (arousal)
    if a > 0.3:
        energy = "energetically"
    elif a < -0.3:
        energy = "calmly"
    else:
        energy = "moderately"
    # Determine control (dominance)
    if d > 0.3:
        control = "and in control"
    elif d < -0.3:
        control = "and overwhelmed"
    else:
        control = "with balance"
    return f"{energy} {valence} {control}"

def get_top_dominant_emotions(current_pad, emotion_map, top_n=3):
    """
    Compares the current PAD vector to each emotion in the mapping (using Euclidean distance)
    and returns the top_n emotions that are closest.
    
    Returns:
        A list of (emotion, distance) tuples.
    """
    similarities = []
    for emotion, vec in emotion_map.items():
        dist = np.linalg.norm(np.array(current_pad) - np.array(vec))
        similarities.append((emotion, dist))
    similarities.sort(key=lambda x: x[1])
    return similarities[:top_n]

# ------------------------------------------------------------------------------
# Virtual Human Class with Personality Biasing
# ------------------------------------------------------------------------------

# The VirtualHuman class represents an agent (Viktor) whose personality biases his emotional reactions.
# Personality is defined by the Big Five traits and is mapped to a baseline PAD vector.
# The personality_bias parameter controls how much the personality influences the immediate reaction.

PADVector = List[float]
BigFive = Dict[str, float]

class VirtualHuman:
    def __init__(self, personality: BigFive, personality_bias: float = 0.3) -> None:
        self.personality = personality
        self.personality_bias = personality_bias

    def bigfive_to_PAD(self) -> PADVector:
        """
        Maps the Big Five personality traits to an approximate PAD vector using a fixed mapping matrix.
        
        Returns:
            A PAD vector reflecting Viktor's personality.
        """
        bigfive_vector = np.array([
            self.personality.get("extraversion", 0.5),
            self.personality.get("neuroticism", 0.5),
            self.personality.get("openness", 0.5),
            self.personality.get("agreeableness", 0.5),
            self.personality.get("conscientiousness", 0.5)
        ])
        # Fixed mapping matrix: rows represent (Pleasure, Arousal, Dominance) dimensions.
        M = np.array([
            [ 0.4, -0.5,  0.0,  0.3,  0.3],
            [ 0.5,  0.5,  0.0,  0.0,  0.0],
            [ 0.4, -0.4,  0.3,  0.0,  0.3]
        ])
        pad = M.dot(bigfive_vector)
        return np.clip(pad, -1, 1).tolist()

    def compute_immediate_reaction(self, user_emotion: PADVector) -> PADVector:
        """
        Computes the immediate reaction by blending the user's detected emotion with the personality-derived PAD.
        
        The blending is controlled by personality_bias:
            Reaction = (1 - personality_bias) * (user emotion) + personality_bias * (personality PAD)
        
        Returns:
            The personality-biased emotion vector.
        """
        personality_PAD = np.array(self.bigfive_to_PAD())
        user_vec = np.array(user_emotion)
        final_vec = (1 - self.personality_bias) * user_vec + self.personality_bias * personality_PAD
        return np.clip(final_vec, -1, 1).tolist()

    def generate_response(self, final_emotion: PADVector) -> str:
        """
        Converts the final PAD vector into a natural language response.
        
        Returns:
            A string describing Viktor's emotional state.
        """
        p, a, d = final_emotion
        if p >= 0.5:
            sentiment = "ecstatic"
        elif p >= 0.2:
            sentiment = "happy"
        elif p >= -0.2:
            sentiment = "neutral"
        elif p >= -0.5:
            sentiment = "downcast"
        else:
            sentiment = "miserable"
        if a >= 0.5:
            energy = "with high energy"
        elif a >= 0.2:
            energy = "energetically"
        elif a >= -0.2:
            energy = "calmly"
        elif a >= -0.5:
            energy = "with low energy"
        else:
            energy = "lethargically"
        if d >= 0.5:
            control = "feeling dominant"
        elif d >= 0.2:
            control = "in control"
        elif d >= -0.2:
            control = "balanced"
        elif d >= -0.5:
            control = "submissive"
        else:
            control = "overwhelmed"
        return f"I feel {sentiment}, {energy} and {control}."

# ------------------------------------------------------------------------------
# Dynamic Alpha Calculation
# ------------------------------------------------------------------------------

def compute_dynamic_alpha(current_mood, new_emotion, base_alpha=BASE_ALPHA):
    """
    Computes an update factor (alpha) based on the cosine similarity between the current mood
    and the new (personality-biased) emotion vector.
    
    A higher cosine similarity (i.e. vectors aligned) yields a larger alpha, meaning a stronger update.
    
    Returns:
        The computed alpha value.
    """
    current = np.array(current_mood)
    new = np.array(new_emotion)
    norm_current = np.linalg.norm(current)
    norm_new = np.linalg.norm(new)
    if norm_current == 0 or norm_new == 0:
        return base_alpha
    cos_sim = np.dot(current, new) / (norm_current * norm_new)
    # Map cosine similarity (-1 to 1) to a new alpha between 0.1 and 0.4 (example mapping).
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
    
    Args:
        ax: The matplotlib 3D axes.
        mood: The current mood state.
        biased_emotion: The last computed personality-biased emotion vector.
    """
    ax.cla()  # Clear the previous plot.
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('Pleasure')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    ax.set_title("Current Mood & Personality-Biased Emotion in PAD Space")
    
    # Plot each emotion's PAD vector as a gray marker with a label.
    for emotion, coord in advanced_emotion_to_PAD.items():
        ax.scatter(coord[0], coord[1], coord[2], c='gray', marker='^', s=50)
        ax.text(coord[0], coord[1], coord[2], emotion, size=8, color='black')
    
    # Plot the current global mood (red dot).
    ax.scatter(mood[0], mood[1], mood[2], c='r', marker='o', s=100, label='Global Mood')
    
    # If available, plot the personality-biased emotion vector as an arrow from the origin.
    if biased_emotion is not None:
        ax.quiver(0, 0, 0, biased_emotion[0], biased_emotion[1], biased_emotion[2],
                  color='b', arrow_length_ratio=0.1, label='Biased Emotion Vector')
    ax.legend()
    plt.draw()

# ------------------------------------------------------------------------------
# Timer Callback for Continuous Decay
# ------------------------------------------------------------------------------

def decay_callback():
    """
    Callback function called by a matplotlib timer every 1000 ms.
    It applies decay to the current mood state to gradually pull it back toward neutral,
    then updates the plot.
    """
    global mood_state
    mood_state = [val * DECAY for val in mood_state]
    update_plot(ax, mood_state, last_biased_emotion)

# ------------------------------------------------------------------------------
# Background Input Thread for User Interaction
# ------------------------------------------------------------------------------

def process_input():
    """
    Runs in a background thread to continuously read user input from the console.
    For each input, it:
      - Detects the composite emotion using the classifier.
      - Applies personality bias to compute an immediate reaction.
      - Computes a dynamic update factor (alpha) based on cosine similarity.
      - Updates the global mood state.
      - Prints details to the console.
      - Updates the 3D plot.
    
    The loop continues until the user types "exit".
    """
    global mood_state, last_biased_emotion
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        # Detect emotion and compute composite PAD.
        composite_PAD, predictions = detect_emotion_advanced(user_input)
        print("\nDetected Emotion Predictions:")
        top_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)[:3]
        for pred in top_preds:
            print(f"  {pred['label']} : {pred['score']:.2f}")
        print(f"Composite Emotion PAD vector: {composite_PAD}")
        
        # Compute the personality-biased reaction.
        biased_emotion = vh.compute_immediate_reaction(composite_PAD)
        last_biased_emotion = biased_emotion  # Update global record.
        print(f"Personality-biased Emotion PAD vector: {biased_emotion}")
        
        # Compute dynamic update factor (alpha) based on the alignment between current mood and biased emotion.
        dynamic_alpha = compute_dynamic_alpha(mood_state, biased_emotion, base_alpha=BASE_ALPHA)
        print(f"Dynamic update factor (alpha): {dynamic_alpha:.2f}")
        
        # Update the global mood state.
        mood_state = update_mood(mood_state, biased_emotion, alpha=dynamic_alpha)
        print(f"Updated Global Mood (PAD): {mood_state} -> {mood_to_description(mood_state)}")
        
        # Generate and print Viktor's response.
        response = vh.generate_response(mood_state)
        print("Viktor AI Response:", response)
        
        # Display the top three dominant emotions.
        dominant_emotions = get_top_dominant_emotions(mood_state, advanced_emotion_to_PAD, top_n=3)
        print("Dominant Emotions:")
        for emotion, dist in dominant_emotions:
            explanation = emotion_descriptions.get(emotion, "No description available.")
            print(f"  {emotion.capitalize()} (distance: {dist:.2f}): {explanation}")
        
        # Update the plot with the new mood state.
        update_plot(ax, mood_state, last_biased_emotion)

# ------------------------------------------------------------------------------
# Main Setup and Event Loop
# ------------------------------------------------------------------------------

# Create the matplotlib figure and 3D axes.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a timer (runs in the main thread) to apply decay every 1000 ms.
timer = fig.canvas.new_timer(interval=1000)
timer.add_callback(decay_callback)
timer.start()

# Define Viktor's personality using the Big Five traits.
# These values define how much the personality will bias the detected emotion.
personality: BigFive = {
    "extraversion": 0.3,
    "neuroticism": 0.4,
    "openness": 0.9,
    "agreeableness": 0.5,
    "conscientiousness": 0.9
}
# Create the VirtualHuman (Viktor) with a personality bias parameter.
vh = VirtualHuman(personality=personality, personality_bias=0.4)

print("Viktor AI Emotion Reaction System. Type 'exit' to quit.")
# Initial plot update.
update_plot(ax, mood_state, last_biased_emotion)

# Start the background thread for user input (without daemon flag so the program stays alive).
input_thread = threading.Thread(target=process_input)
input_thread.start()

# Block the main thread with the GUI event loop.
plt.show(block=True)
