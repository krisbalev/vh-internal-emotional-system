import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# Download VADER lexicon (only runs the first time)
nltk.download('vader_lexicon')


# Global mood state represented as a PAD vector (pleasure, arousal, dominance)
mood_state = [0.0, 0.0, 0.0]  # starting at neutral

# Define a constant update factor for mood changes
ALPHA = 0.2  # Adjust to tune sensitivity

# Full mapping of 24 emotions (from the paper's Table 10.3) to PAD vectors
emotion_to_PAD = {
    "admiration":      (0.5, 0.3, -0.2),
    "anger":           (-0.51, 0.59, 0.25),
    "disliking":       (-0.4, 0.2, 0.1),
    "disappointment":  (-0.3, 0.1, -0.4),
    "distress":        (-0.4, -0.2, 0.5),
    "fear":            (-0.64, 0.60, -0.43),
    "fears-confirmed": (-0.5, -0.3, -0.7),
    "gloating":        (0.3, -0.3, -0.1),
    "gratification":   (0.6, 0.5, 0.4),
    "gratitude":       (0.4, 0.2, -0.3),
    "happy for":       (0.4, 0.2, 0.2),
    "satisfaction":    (0.3, -0.2, 0.4),
    "hate":            (-0.6, 0.6, 0.3),
    "hope":            (0.2, 0.2, -0.1),
    "joy":             (0.4, 0.2, 0.1),
    "liking":          (0.40, 0.16, -0.24),
    "love":            (0.3, 0.1, 0.2),
    "pity":            (-0.4, -0.2, -0.5),
    "pride":           (0.4, 0.3, 0.3),
    "relief":          (0.2, -0.3, 0.4),
    "remorse":         (-0.3, 0.1, -0.6),
    "reproach":        (-0.3, -0.1, 0.4),
    "resentment":      (-0.2, -0.3, -0.2),
    "shame":           (-0.3, 0.1, -0.6)
}

# Define keywords for each emotion to aid detection
emotion_keywords = {
    "admiration":      ["admire", "admiration"],
    "anger":           ["angry", "mad", "furious", "irate"],
    "disliking":       ["dislike", "detest", "loathe"],
    "disappointment":  ["disappointed", "disappointment"],
    "distress":        ["distressed", "distress", "troubled"],
    "fear":            ["scared", "fear", "frightened", "terrified"],
    "fears-confirmed": ["fears confirmed", "fear confirmed"],
    "gloating":        ["gloating", "smug"],
    "gratification":   ["gratification"],
    "gratitude":       ["grateful", "gratitude"],
    "happy for":       ["happy for"],
    "satisfaction":    ["satisfaction", "satisfied", "content"],
    "hate":            ["hate", "hated"],
    "hope":            ["hope", "optimistic"],
    "joy":             ["joy", "joyful", "happy"],
    "liking":          ["like", "liking"],
    "love":            ["love", "loving"],
    "pity":            ["pity", "pitiable"],
    "pride":           ["proud", "pride"],
    "relief":          ["relief", "relieved"],
    "remorse":         ["remorse", "regret"],
    "reproach":        ["reproach", "blame"],
    "resentment":      ["resent", "resentment"],
    "shame":           ["shame", "shamed"]
}

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

def detect_emotion(text):
    """
    Detect emotion from text using VADER sentiment analysis combined with keyword heuristics.
    Returns one of the 24 emotions if a keyword is found; otherwise, falls back to VADER-based classification.
    """
    text_lower = text.lower()
    
    # Check if any emotion keyword is present
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                return emotion
    
    # Fallback using VADER compound score
    scores = sid.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.3:
        return "joy"
    elif compound <= -0.3:
        # Default to anger for strong negative sentiment when no keyword is found
        return "anger"
    else:
        return "neutral"

def update_mood(current_mood, emotion_PAD, alpha=ALPHA):
    """
    Update the mood state by moving a fraction (alpha) towards the new emotion's PAD vector.
    """
    new_mood = [
        current_mood[0] + alpha * emotion_PAD[0],
        current_mood[1] + alpha * emotion_PAD[1],
        current_mood[2] + alpha * emotion_PAD[2]
    ]
    # Clamp values between -1 and 1 for stability.
    new_mood = [max(-1, min(1, val)) for val in new_mood]
    return new_mood

def mood_to_description(mood):
    """
    Convert the numeric mood state (PAD vector) to a simple descriptive string based on the pleasure component.
    """
    p, a, d = mood
    if p > 0.3:
        return "happy"
    elif p < -0.3:
        return "sad"
    else:
        return "neutral"
    
def main():
    global mood_state

    # Set up interactive 3D plot for visualization
    plt.ion()  # Enable interactive mode
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('Pleasure')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    plt.title("Current Mood and Detected Emotion in PAD Space")

    print("Virtual Human Prototype. Type 'exit' to quit.")
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Step 1: Detect emotion from input
        detected_emotion = detect_emotion(user_input)
        print(f"Detected emotion: {detected_emotion}")

        # Step 2: Get PAD vector for detected emotion
        pad_vector = emotion_to_PAD.get(detected_emotion, (0.0, 0.0, 0.0))
        print(f"Emotion PAD vector: {pad_vector}")

        # Step 3: Update mood state
        mood_state = update_mood(mood_state, pad_vector)
        mood_desc = mood_to_description(mood_state)
        print(f"Updated Mood (PAD): {mood_state}  -> {mood_desc}")

        # Step 4: Update the 3D visualization
        ax.cla()  # Clear the current axes
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('Pleasure')
        ax.set_ylabel('Arousal')
        ax.set_zlabel('Dominance')
        ax.set_title("Current Mood and Detected Emotion in PAD Space")
        
        # Plot current mood as a red dot
        ax.scatter(mood_state[0], mood_state[1], mood_state[2], c='r', marker='o', s=100, label='Current Mood')
        # Plot the detected emotion vector as a blue arrow from the origin
        ax.quiver(0, 0, 0, pad_vector[0], pad_vector[1], pad_vector[2], color='b', arrow_length_ratio=0.1, label='Emotion Vector')
        ax.legend()
        
        plt.draw()
        plt.pause(0.1)
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
