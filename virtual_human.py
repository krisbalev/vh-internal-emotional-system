# virtual_human.py
import numpy as np
from emotion_processing import compute_emotion_weights
from mood import get_top_dominant_emotions
from config import advanced_emotion_to_PAD
from chatgpt_client import generate_chatgpt_response

class VirtualHuman:
    def __init__(self, personality, personality_bias=0.4, mood_bias=0.2):
        """
        personality: A dictionary with Big Five scores.
        personality_bias and mood_bias define the blending weights.
        """
        self.personality = personality
        self.personality_bias = personality_bias
        self.mood_bias = mood_bias
        # Compute personality baseline in PAD space.
        self.personality_PAD = self.bigfive_to_PAD()
        optimal_weights, emotion_names = compute_emotion_weights(self.personality_PAD, advanced_emotion_to_PAD, lambda_val=0.01)
        self.optimal_weights = {name: weight for name, weight in zip(emotion_names, optimal_weights)}
    
    def bigfive_to_PAD(self):
        """
        Maps the Big Five traits to a PAD vector.
        """
        extraversion = self.personality.get("extraversion", 0.5)
        agreeableness = self.personality.get("agreeableness", 0.5)
        neuroticism = self.personality.get("neuroticism", 0.5)
        conscientiousness = self.personality.get("conscientiousness", 0.5)
        # Convert Neuroticism to emotional stability:
        emotional_stability = 1 - neuroticism
        sophistication = self.personality.get("openness", 0.5)
        
        P = 0.59 * agreeableness + 0.19 * emotional_stability + 0.21 * extraversion
        A = -0.57 * emotional_stability + 0.30 * agreeableness + 0.15 * sophistication
        D = 0.60 * extraversion - 0.32 * agreeableness + 0.25 * sophistication + 0.17 * conscientiousness
        return [np.clip(P, -1, 1), np.clip(A, -1, 1), np.clip(D, -1, 1)]
    
    def compute_immediate_reaction(self, user_emotion, current_mood):
        """
        Blends the raw weighted emotion with the current mood.
        """
        user_vec = np.array(user_emotion)
        current_mood_vec = np.array(current_mood)
        final_vec = (1 - self.mood_bias) * user_vec + self.mood_bias * current_mood_vec
        return np.clip(final_vec, -1, 1).tolist()
    
    def generate_response(self, user_text, final_emotion):
        """
        Generates a ChatGPT response based on the user input and updated emotion.
        """
        dominant_emotions_data = get_top_dominant_emotions(final_emotion, advanced_emotion_to_PAD, top_n=3)
        dominant_emotion_names = [emotion for emotion, _ in dominant_emotions_data]
        return generate_chatgpt_response(user_text, final_emotion, dominant_emotion_names, self.personality)
