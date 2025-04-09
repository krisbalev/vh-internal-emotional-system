# emotion_processing.py
from transformers import pipeline
import numpy as np
from scipy.optimize import minimize
from config import advanced_emotion_to_PAD

# Initialize the emotion classifier from Hugging Face
classifier = pipeline(
    task="text-classification", 
    model="SamLowe/roberta-base-go_emotions",  # Model for emotion detection
    top_k=None
)

def compute_emotion_weights(personality, emotion_mapping, lambda_val=0.01):
    """
    Solves a regularized least squares problem to compute emotion weights based on the personality (PAD vector).
    """
    emotion_names = list(emotion_mapping.keys())
    S = len(emotion_names)
    # Build the PAD matrix (3 x S)
    D = np.array([emotion_mapping[emo] for emo in emotion_names]).T
    P = np.array(personality)
    
    def objective(w):
        return np.linalg.norm(D.dot(w) - P)**2 + lambda_val * np.linalg.norm(w)**2
    
    w0 = np.ones(S)
    bounds = [(0, None)] * S
    res = minimize(objective, w0, bounds=bounds)
    
    if res.success:
        optimal_weights = res.x
    else:
        raise ValueError("Optimization did not converge.")
    
    return optimal_weights, emotion_names

def detect_emotion_weighted(text, optimal_weights):
    """
    Uses the classifier to predict emotions from text and computes a weighted composite PAD vector.
    """
    predictions = classifier(text)[0]
    composite_PAD = [0.0, 0.0, 0.0]
    total_weighted_score = 0.0
    for pred in predictions:
        label = pred["label"].lower()  # Normalize to match keys in mapping
        score = pred["score"]
        weight = optimal_weights.get(label, 1.0)
        total_weighted_score += score * weight
        pad = advanced_emotion_to_PAD.get(label, (0.0, 0.0, 0.0))
        composite_PAD[0] += score * weight * pad[0]
        composite_PAD[1] += score * weight * pad[1]
        composite_PAD[2] += score * weight * pad[2]
    if total_weighted_score > 0:
        composite_PAD = [x / total_weighted_score for x in composite_PAD]
    return composite_PAD, predictions
