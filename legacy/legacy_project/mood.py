import numpy as np
import matplotlib.pyplot as plt
from config import DECAY_RATE, advanced_emotion_to_PAD

def update_mood(current_mood, biased_emotion, alpha):
    """
    Update the current mood via exponential smoothing.
    """
    new_mood = [(1 - alpha) * c + alpha * b for c, b in zip(current_mood, biased_emotion)]
    return [max(-1, min(1, val)) for val in new_mood]

def mood_to_description(mood):
    """
    Convert a PAD mood vector to a natural-language description.
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

def compute_dynamic_alpha(current_mood, new_emotion, base_alpha=0.2):
    """
    Adjust the update rate based on the cosine similarity of mood and new emotion.
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

def get_top_dominant_emotions(current_pad, emotion_map, top_n=3):
    """
    Returns the top N emotions (by Euclidean distance) closest to the current mood.
    """
    similarities = []
    for emotion, vec in emotion_map.items():
        dist = np.linalg.norm(np.array(current_pad) - np.array(vec))
        similarities.append((emotion, dist))
    similarities.sort(key=lambda x: x[1])
    return similarities[:top_n]

def update_plot(ax, mood, biased_emotion):
    """
    Updates a 3D plot with static emotion positions, current mood, and the biased emotion vector.
    """
    ax.cla()  # Clear previous content
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

def decay_callback(mood_state, personality_baseline, ax, last_biased_emotion):
    """
    Apply exponential decay to the current mood, nudging it toward the personality baseline.
    """
    mood_state[:] = [current + DECAY_RATE * (baseline - current)
                     for current, baseline in zip(mood_state, personality_baseline)]
    update_plot(ax, mood_state, last_biased_emotion)
