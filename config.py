import numpy as np

# Emotion labels and PAD space directions
emotion_labels = [
    "Hope", "Gratitude", "Admiration", "Gratification", "HappyFor", "Joy", "Love",
    "Pride", "Relief", "Satisfaction", "Gloating", "Remorse", "Disappointment",
    "Fear", "Shame", "Resentment", "Fears-confirmed", "Pity", "Distress",
    "Anger", "Hate", "Reproach"
]

D = np.array([
    [0.2,  0.2,  -0.1], # Hope
    [0.4,  0.2,  -0.3], # Gratitude
    [0.5,  0.3,  -0.2], # Admiration
    [0.6,  0.5,   0.4], # Gratification
    [0.4,  0.2,   0.2], # HappyFor
    [0.4,  0.2,   0.1], # Joy
    [0.3,  0.1,   0.2], # Love
    [0.4,  0.3,   0.3], # Pride
    [0.2, -0.3,   0.4], # Relief
    [0.3, -0.2,   0.4], # Satisfaction
    [0.3, -0.3,  -0.1], # Gloating
    [-0.3,  0.1, -0.6], # Remorse
    [-0.3,  0.1, -0.4], # Disappointment
    [-0.64, 0.60, -0.43], # Fear
    [-0.3,  0.1, -0.6], # Shame
    [-0.2, -0.3, -0.2], # Resentment
    [-0.5, -0.3, -0.7], # Fears-confirmed 
    [-0.4, -0.2, -0.5], # Pity
    [-0.4, -0.2, -0.5], # Distress
    [-0.51, 0.59,  0.25], # Anger
    [-0.6,  0.6,   0.3], # Hate
    [-0.3, -0.1,  0.4] # Reproach
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
