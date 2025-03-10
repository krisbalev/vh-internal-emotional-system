import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from transformers import pipeline
import torch

# Initialize the transformer-based emotion classifier using the GoEmotions model.
classifier = pipeline(
    "text-classification",
    model="monologg/bert-base-cased-goemotions-original",
    top_k=None,
    truncation=True
)

# Global mood state represented as a PAD vector (pleasure, arousal, dominance)
mood_state = [0.0, 0.0, 0.0]  # starting at neutral

# Define a constant update factor for mood changes
ALPHA = 0.2  # Adjust to tune sensitivity

# Mapping from GoEmotions labels (27 emotions) to PAD vectors.
# The values below are approximate and inspired by psychological mappings.
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

def detect_emotion_advanced(text):
    """
    Uses the transformer-based classifier to detect a broad range of emotions.
    Aggregates predictions from all 27 labels weighted by their scores to compute a composite PAD vector.
    Returns the composite PAD vector along with the raw predictions.
    """
    predictions = classifier(text)[0]
    
    composite_PAD = [0.0, 0.0, 0.0]
    total_score = 0.0
    for pred in predictions:
        label = pred["label"].lower()  # normalize label to lowercase
        score = pred["score"]
        total_score += score
        pad = advanced_emotion_to_PAD.get(label, (0.0, 0.0, 0.0))
        composite_PAD[0] += score * pad[0]
        composite_PAD[1] += score * pad[1]
        composite_PAD[2] += score * pad[2]
    
    # Normalize to get an average PAD vector if total_score > 0
    if total_score > 0:
        composite_PAD = [x / total_score for x in composite_PAD]
    
    return composite_PAD, predictions

def update_mood(current_mood, emotion_PAD, alpha=ALPHA):
    """
    Update the global mood state by moving it a fraction (alpha) towards the provided emotion PAD vector.
    """
    new_mood = [
        current_mood[0] + alpha * emotion_PAD[0],
        current_mood[1] + alpha * emotion_PAD[1],
        current_mood[2] + alpha * emotion_PAD[2]
    ]
    # Clamp each value between -1 and 1.
    new_mood = [max(-1, min(1, val)) for val in new_mood]
    return new_mood

def mood_to_description(mood):
    """
    Converts the numeric mood state (PAD vector) to a nuanced descriptive string.
    Considers pleasure, arousal, and dominance:
      - Pleasure indicates overall valence (happy vs. sad).
      - Arousal indicates energy (energetic vs. calm).
      - Dominance indicates control (in control vs. overwhelmed).
    """
    p, a, d = mood

    # Determine valence from pleasure
    if p > 0.3:
        valence = "happy"
    elif p < -0.3:
        valence = "sad"
    else:
        valence = "neutral"

    # Determine energy level from arousal
    if a > 0.3:
        energy = "energetically"
    elif a < -0.3:
        energy = "calmly"
    else:
        energy = "moderately"

    # Determine control from dominance
    if d > 0.3:
        control = "and in control"
    elif d < -0.3:
        control = "and overwhelmed"
    else:
        control = "with balance"

    return f"{energy} {valence} {control}"

def main():
    global mood_state

    # Set up interactive 3D plot for visualization
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('Pleasure')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    plt.title("Current Mood and Composite Emotion in PAD Space")

    print("Advanced Multi-Label Emotion Detection Prototype. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Detect emotions and compute composite PAD vector.
        composite_PAD, predictions = detect_emotion_advanced(user_input)
        # Display top 3 predicted emotions for reference.
        top_preds = sorted(predictions, key=lambda x: x["score"], reverse=True)[:3]
        print("Top Predictions:")
        for pred in top_preds:
            print(f"  {pred['label']} : {pred['score']:.2f}")
        print(f"Composite Emotion PAD vector: {composite_PAD}")

        # Update the global mood state.
        mood_state = update_mood(mood_state, composite_PAD)
        mood_desc = mood_to_description(mood_state)
        print(f"Updated Mood (PAD): {mood_state}  -> {mood_desc}")

        # Update the 3D visualization.
        ax.cla()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('Pleasure')
        ax.set_ylabel('Arousal')
        ax.set_zlabel('Dominance')
        ax.set_title("Current Mood and Composite Emotion in PAD Space")
        
        # Plot current mood as a red dot.
        ax.scatter(mood_state[0], mood_state[1], mood_state[2], c='r', marker='o', s=100, label='Current Mood')
        # Plot composite emotion vector as a blue arrow from the origin.
        ax.quiver(0, 0, 0, composite_PAD[0], composite_PAD[1], composite_PAD[2],
                  color='b', arrow_length_ratio=0.1, label='Composite Emotion Vector')
        ax.legend()
        
        plt.draw()
        plt.pause(0.1)
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
