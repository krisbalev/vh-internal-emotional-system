import numpy as np

# Emotion labels and PAD space directions
emotion_labels = [
    "Hope", "Gratitude", "Admiration", "Gratification", "HappyFor", "Joy", "Love",
    "Pride", "Relief", "Satisfaction", "Gloating", "Remorse", "Disappointment",
    "Fear", "Shame", "Resentment", "Fears-confirmed", "Pity", "Distress",
    "Anger", "Hate", "Reproach", "Amusement", "Annoyance", "Approval", "Caring", "Confusion", "Curiosity",
    "Desire", "Disapproval", "Disgust", "Embarrassment", "Excitement", "Grief",
    "Nervousness", "Optimism", "Realization", "Sadness", "Surprise"
]

D = np.array([
    [ 0.22,  0.28, -0.23],  # Hope
    [ 0.69, -0.09,  0.05],  # Gratitude
    [ 0.49, -0.19,  0.05],  # Admiration
    [ 0.39, -0.18,  0.41],  # Gratification
    [ 0.75,  0.17,  0.37],  # HappyFor
    [ 0.82,  0.43,  0.55],  # Joy
    [ 0.80,  0.14,  0.30],  # Love
    [ 0.72,  0.20,  0.57],  # Pride
    [ 0.73, -0.24,  0.06],  # Relief
    [ 0.65, -0.42,  0.35],  # Satisfaction
    [ 0.08,  0.11,  0.44],  # Gloating
    [-0.42, -0.01, -0.35],  # Remorse
    [-0.64, -0.17, -0.41],  # Disappointment
    [-0.74,  0.47, -0.62],  # Fear
    [-0.66,  0.05, -0.63],  # Shame
    [-0.52,  0.00,  0.03],  # Resentment
    [-0.74,  0.42, -0.52],  # Fears-confirmed
    [-0.27, -0.24,  0.24],  # Pity
    [-0.75, -0.31, -0.47],  # Distress
    [-0.62,  0.59,  0.23],  # Anger
    [-0.52,  0.00,  0.28],  # Hate
    [-0.41,  0.47,  0.50],  # Reproach
    [ 0.45,  0.25,  0.00],  # Amusement
    [-0.40,  0.40,  0.10],  # Annoyance
    [ 0.30,  0.10,  0.00],  # Approval
    [ 0.40,  0.10,  0.20],  # Caring
    [ 0.00,  0.00, -0.10],  # Confusion
    [ 0.20,  0.30,  0.00],  # Curiosity
    [ 0.40,  0.50,  0.20],  # Desire
    [-0.30,  0.20,  0.00],  # Disapproval
    [-0.40,  0.20,  0.10],  # Disgust
    [-0.20, -0.10, -0.10],  # Embarrassment
    [ 0.60,  0.70,  0.10],  # Excitement
    [-0.50, -0.40, -0.20],  # Grief
    [-0.20,  0.30, -0.20],  # Nervousness
    [ 0.40,  0.30,  0.00],  # Optimism
    [ 0.10,  0.10,  0.00],  # Realization
    [-0.40, -0.20, -0.30],  # Sadness
    [ 0.00,  0.60,  0.00],  # Surprise
])

S = len(emotion_labels)

# Initial emotion probabilities
p = np.ones(S) / S

# Big Five to PAD conversion
Q = np.array([
    [0.00,  0.00,  0.21,  0.59,  0.19],
    [0.15,  0.00,  0.00,  0.30, -0.57],
    [0.25,  0.17,  0.60, -0.32,  0.00]
])
F = np.array([0.9, 0.9, 0.3, 0.5, 0.4])
# P = Q @ F   # Personality point in PAD
P = D[20].copy() # Baseline angry character

# Model parameters
alpha = 2.0
mu_P = 0.1
lambda_e = 1.0
lambda_m = 0.001
event_rate = 1/3.0

# Derived parameter
a = alpha / lambda_e * (1 - np.exp(-lambda_e / event_rate))
TARGET_SHIFT = 0.5

# Simulation timestep
dt = 0.1

def explain_personality(P, 
                        dim_names=("Pleasure", "Arousal", "Dominance"),
                        thresholds=(0.5, 0.2)):
    """
    Explain a PAD-vector personality P in words.
    
    Args:
      P: array-like of length 3: [Pleasure, Arousal, Dominance], each in [-1,1].
      dim_names: names for each dimension.
      thresholds: (high_thresh, low_thresh) for strong vs. mild.
    
    Returns:
      A multi-sentence string describing the personality.
    """
    high, low = thresholds
    descriptions = {
        "Pleasure": {
            "high":  "very positive, friendly, and approachable",
            "mid":   "somewhat positive and pleasant",
            "neu":   "fairly neutral or even-keeled",
            "midn":  "somewhat negative or irritable",
            "lown":  "quite negative, unfriendly, or hostile"
        },
        "Arousal": {
            "high":  "highly energetic, excitable, and restless",
            "mid":   "moderately alert and engaged",
            "neu":   "calm or balanced in energy",
            "midn":  "somewhat lethargic or subdued",
            "lown":  "very low-energy, sleepy, or withdrawn"
        },
        "Dominance": {
            "high":  "strongly assertive, controlling, and confident",
            "mid":   "somewhat assertive and self-reliant",
            "neu":   "balanced between following and leading",
            "midn":  "somewhat yielding or indecisive",
            "lown":  "very submissive or deferential"
        }
    }
    
    parts = []
    for val, name in zip(P, dim_names):
        if val >= high:
            lvl = "high"
        elif val >= low:
            lvl = "mid"
        elif val > -low:
            lvl = "neu"
        elif val > -high:
            lvl = "midn"
        else:
            lvl = "lown"
        desc = descriptions[name][lvl]
        parts.append(f"• **{name}** = {val:.2f}: {desc}.")

    return parts

P_full = explain_personality(P)
