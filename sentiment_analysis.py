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
        # Convert Neuroticism to Emotional Stability:
        emotional_stability = 1 - neuroticism
        sophistication = self.personality.get("openness", 0.5)  # treating openness as sophistication

        # Apply Mehrabian's regression equations:
        P = 0.59 * agreeableness + 0.25 * emotional_stability + 0.19 * extraversion
        A = -0.65 * emotional_stability + 0.42 * agreeableness
        D = 0.77 * extraversion - 0.27 * agreeableness + 0.21 * sophistication

        # Optionally, clip values to [-1, 1] if needed.
        return [np.clip(P, -1, 1), np.clip(A, -1, 1), np.clip(D, -1, 1)]

    def compute_immediate_reaction(self, user_emotion: PADVector, current_mood: PADVector) -> PADVector:
        """
        Computes the immediate emotion reaction by blending:
          - The raw emotion (from user input)
          - The personality-based baseline (from Big Five)
          - The current mood (as an appraisal bias)
          
        The blending weights are:
          (1 - personality_bias - mood_bias) * user_emotion +
          personality_bias * personality_PAD +
          mood_bias * current_mood
        """
        # Get personality baseline from Big Five mapping
        personality_PAD = np.array(self.bigfive_to_PAD())
        user_vec = np.array(user_emotion)
        current_mood_vec = np.array(current_mood)
        # Calculate weight for raw emotion: total weight must sum to 1.
        weight_user = 1 - self.personality_bias - self.mood_bias
        # Blend the three components: raw emotion, personality, and current mood.
        final_vec = weight_user * user_vec + self.personality_bias * personality_PAD + self.mood_bias * current_mood_vec
        return np.clip(final_vec, -1, 1).tolist()


    def generate_response(self, final_emotion: PADVector) -> str:
        """
        Converts the final PAD vector into a natural language description.
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
    """
    current = np.array(current_mood)
    new = np.array(new_emotion)
    norm_current = np.linalg.norm(current)
    norm_new = np.linalg.norm(new)
    if norm_current == 0 or norm_new == 0:
        return base_alpha
    # Compute cosine similarity.
    cos_sim = np.dot(current, new) / (norm_current * norm_new)
    # Map cosine similarity to alpha value.
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
    # Compute personality baseline from Big Five mapping.
    personality_baseline = vh.bigfive_to_PAD()
    # Update each component: mood = mood + DECAY_RATE * (baseline - mood)
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
        composite_PAD, predictions = detect_emotion_advanced(user_input)
        print("\nDetected Emotion Predictions:")
        top_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)[:3]
        for pred in top_preds:
            print(f"  {pred['label']} : {pred['score']:.2f}")
        print(f"Composite Emotion PAD vector: {composite_PAD}")
        
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
        
        # Generate and print Viktor's response.
        response = vh.generate_response(mood_state)
        print("Viktor AI Response:", response)
        
        # Display the top three dominant emotions.
        dominant_emotions = get_top_dominant_emotions(mood_state, advanced_emotion_to_PAD, top_n=3)
        print("Dominant Emotions:")
        for emotion, dist in dominant_emotions:
            explanation = emotion_descriptions.get(emotion, "No description available.")
            print(f"  {emotion.capitalize()} (distance: {dist:.2f}): {explanation}")
        
        update_plot(ax, mood_state, last_biased_emotion)

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

plt.show(block=True)
