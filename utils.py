import numpy as np

# Compute emotion intensity with personality/mood bias
def calculate_emotion_intensity(I0, current_mood, personality, emotion_direction, target_shift=0.5):
    """
    Calculate the emotion intensity by projecting personality and mood
    onto the emotion's PAD direction, normalizing via cosine, then
    scaling by target_shift.

    Args:
        I0 (float): Base intensity (from OCC appraisal).
        current_mood (np.array): M in PAD space.
        personality (np.array): P in PAD space.
        emotion_direction (np.array): D_i in PAD space.
        target_shift (float): Maximum absolute bias contribution.
    
    Returns:
        float: Clipped intensity in [0, 1].
    """
    # normalize so dot‐product becomes cosine similarity in [-1,1]
    norm_P = np.linalg.norm(personality)
    norm_M = np.linalg.norm(current_mood)
    norm_D = np.linalg.norm(emotion_direction)
    if norm_P == 0 or norm_M == 0 or norm_D == 0:
        # fallback to no bias if any vector is zero
        return np.clip(I0, 0, 1)

    proj_P = np.dot(personality, emotion_direction) / (norm_P * norm_D)
    proj_M = np.dot(current_mood, emotion_direction) / (norm_M * norm_D)

    # bias terms capped by target_shift
    theta_P_i = target_shift * max(0, proj_P)
    theta_M_i = target_shift * max(0, proj_M)

    raw_intensity = I0 + theta_P_i + theta_M_i
    return np.clip(raw_intensity, 0, 1)