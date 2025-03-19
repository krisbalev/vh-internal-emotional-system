import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from transformers import pipeline
import numpy as np
from typing import Dict, List

# Global constants and variables.
BASE_ALPHA = 0.2  # Base sensitivity factor.
mood_state = [0.0, 0.0, 0.0]  # Initial global mood state (neutral).

# Mapping from GoEmotions labels (27 emotions) to approximate PAD vectors.
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

# Short explanations for each emotion.
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

# Initialize the transformer-based emotion classifier.
classifier = pipeline(
    "text-classification",
    model="monologg/bert-base-cased-goemotions-original",
    top_k=None,
    truncation=True
)

def detect_emotion_advanced(text: str):
    """
    Detects a broad range of emotions using a transformer-based classifier.
    Aggregates predictions weighted by their scores to compute a composite PAD vector.
    Returns the composite PAD vector along with the raw predictions.
    """
    predictions = classifier(text)[0]
    
    composite_PAD = [0.0, 0.0, 0.0]
    total_score = 0.0
    for pred in predictions:
        label = pred["label"].lower()  # Normalize label.
        score = pred["score"]
        total_score += score
        pad = advanced_emotion_to_PAD.get(label, (0.0, 0.0, 0.0))
        composite_PAD[0] += score * pad[0]
        composite_PAD[1] += score * pad[1]
        composite_PAD[2] += score * pad[2]
    
    if total_score > 0:
        composite_PAD = [x / total_score for x in composite_PAD]
    
    return composite_PAD, predictions

def update_mood(current_mood, emotion_PAD, alpha):
    """
    Updates the global mood state by moving it a fraction (alpha) toward the provided emotion PAD vector.
    Clamps each dimension between -1 and 1.
    """
    new_mood = [
        current_mood[0] + alpha * emotion_PAD[0],
        current_mood[1] + alpha * emotion_PAD[1],
        current_mood[2] + alpha * emotion_PAD[2]
    ]
    new_mood = [max(-1, min(1, val)) for val in new_mood]
    return new_mood

def mood_to_description(mood):
    """
    Converts the numeric mood state (PAD vector) to a descriptive string.
    - Pleasure (p) indicates valence.
    - Arousal (a) indicates energy.
    - Dominance (d) indicates the degree of control.
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
    Compares the current PAD vector to each emotion in emotion_map using Euclidean distance.
    Returns the top_n emotions (lowest distance) along with their distances.
    """
    similarities = []
    for emotion, vec in emotion_map.items():
        dist = np.linalg.norm(np.array(current_pad) - np.array(vec))
        similarities.append((emotion, dist))
    similarities.sort(key=lambda x: x[1])
    return similarities[:top_n]

# --- Virtual Human with Personality Biasing ---
# Viktor's personality based on his character details:
# Extraversion: 0.3, Neuroticism: 0.4, Openness: 0.9, Agreeableness: 0.5, Conscientiousness: 0.9

PADVector = List[float]
BigFive = Dict[str, float]

class VirtualHuman:
    def __init__(self, personality: BigFive, personality_bias: float = 0.3) -> None:
        self.personality = personality
        self.personality_bias = personality_bias

    def bigfive_to_PAD(self) -> PADVector:
        """
        Maps the Big Five personality traits to an approximate PAD vector.
        Uses a fixed mapping matrix.
        """
        bigfive_vector = np.array([
            self.personality.get("extraversion", 0.5),
            self.personality.get("neuroticism", 0.5),
            self.personality.get("openness", 0.5),
            self.personality.get("agreeableness", 0.5),
            self.personality.get("conscientiousness", 0.5)
        ])
        # Mapping matrix from Big Five to PAD.
        M = np.array([
            [ 0.4, -0.5,  0.0,  0.3,  0.3],
            [ 0.5,  0.5,  0.0,  0.0,  0.0],
            [ 0.4, -0.4,  0.3,  0.0,  0.3]
        ])
        pad = M.dot(bigfive_vector)
        pad = np.clip(pad, -1, 1)
        return pad.tolist()

    def compute_immediate_reaction(self, user_emotion: PADVector) -> PADVector:
        """
        Computes the immediate emotional reaction by combining the user's detected emotion (PAD)
        with the personality-derived PAD vector.
        """
        personality_PAD = np.array(self.bigfive_to_PAD())
        user_vec = np.array(user_emotion)
        final_vec = (1 - self.personality_bias) * user_vec + self.personality_bias * personality_PAD
        final_vec = np.clip(final_vec, -1, 1)
        return final_vec.tolist()

    def generate_response(self, final_emotion: PADVector) -> str:
        """
        Converts the final PAD vector into a natural language response.
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

    def compute_dynamic_alpha(self) -> float:
        """
        Computes a dynamic update factor (alpha) based on the Big Five personality traits.
        Higher neuroticism and lower conscientiousness result in faster emotional changes.
        The computed alpha is clamped between 0.1 and 0.5.
        """
        neuroticism = self.personality.get("neuroticism", 0.5)
        conscientiousness = self.personality.get("conscientiousness", 0.5)
        dynamic_alpha = BASE_ALPHA * (1 + (neuroticism - conscientiousness))
        dynamic_alpha = max(0.1, min(dynamic_alpha, 0.5))
        return dynamic_alpha

# --- Main Interactive Loop ---

def main():
    global mood_state

    # Set up an interactive 3D plot for visualization.
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('Pleasure')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    plt.title("Current Mood & Personality-Biased Emotion in PAD Space")

    # Define Viktor's personality based on his character traits.
    personality: BigFive = {
        "extraversion": 0.3,      # Reserved and direct
        "neuroticism": 0.4,       # Aware of his health, yet stoic
        "openness": 0.9,          # Highly innovative and curious
        "agreeableness": 0.5,     # Direct and pragmatic
        "conscientiousness": 0.9  # Extremely methodical and determined
    }
    # A slightly higher personality_bias reflects Viktor's strong personal vision.
    vh = VirtualHuman(personality=personality, personality_bias=0.4)
    
    print("Viktor AI Emotion Reaction System. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Step 1: Detect the raw composite emotion from the user input.
        composite_PAD, predictions = detect_emotion_advanced(user_input)
        print("\nDetected Emotion Predictions:")
        top_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)[:3]
        for pred in top_preds:
            print(f"  {pred['label']} : {pred['score']:.2f}")
        print(f"Composite Emotion PAD vector: {composite_PAD}")

        # Step 2: Apply personality biasing to the detected emotion.
        biased_emotion = vh.compute_immediate_reaction(composite_PAD)
        print(f"Personality-biased Emotion PAD vector: {biased_emotion}")

        # Step 3: Compute a dynamic update factor based on Viktor's personality.
        dynamic_alpha = vh.compute_dynamic_alpha()
        print(f"Dynamic update factor (alpha): {dynamic_alpha:.2f}")

        # Step 4: Update the global mood state using the personality-biased emotion.
        mood_state = update_mood(mood_state, biased_emotion, alpha=dynamic_alpha)
        mood_desc = mood_to_description(mood_state)
        print(f"Updated Global Mood (PAD): {mood_state}  -> {mood_desc}")

        # Step 5: Generate Viktor's final response based on the updated global mood.
        response = vh.generate_response(mood_state)
        print("Viktor AI Response:", response)

        # Step 6: Determine and display the top three dominant emotions for the current mood.
        dominant_emotions = get_top_dominant_emotions(mood_state, advanced_emotion_to_PAD, top_n=3)
        print("Dominant Emotions:")
        for emotion, dist in dominant_emotions:
            explanation = emotion_descriptions.get(emotion, "No description available.")
            print(f"  {emotion.capitalize()} (distance: {dist:.2f}): {explanation}")

        # Update the 3D visualization.
        ax.cla()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('Pleasure')
        ax.set_ylabel('Arousal')
        ax.set_zlabel('Dominance')
        ax.set_title("Current Mood & Personality-Biased Emotion in PAD Space")
        
        # Plot the updated global mood as a red dot.
        ax.scatter(mood_state[0], mood_state[1], mood_state[2], c='r', marker='o', s=100, label='Global Mood')
        # Also plot the personality-biased emotion vector as a blue arrow from the origin.
        ax.quiver(0, 0, 0, biased_emotion[0], biased_emotion[1], biased_emotion[2],
                  color='b', arrow_length_ratio=0.1, label='Biased Emotion Vector')
        ax.legend()
        
        plt.draw()
        plt.pause(0.1)
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
