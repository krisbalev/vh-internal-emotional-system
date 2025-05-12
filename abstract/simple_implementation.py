### Lessons learned:
# 1. The model treats the Big5 OCEAN scores in the [-1, 1] range, not [0, 1]
# 2. Many optimal weights from the optimization are 0 meaning the model won't activate for some emotions, 
# but they must be >0, we increase the lower bound to e.g. 0.1 (or a reasonable value)
# 3. The model expects a Poisson event rate to converge to relaxation towards the personality, 
# but since we don't have a Poisson distrubution, we use small virtual events to converge to the personality
# 4. It's not the location in PAD-space that determines the emotion, but the closest angle between the mood vector and the emotion vector
# 5. The model expects zero mood decay (it decays by the Poisson event rate towards the personality), but we can use a small lambda_m to simulate decay towards a neutral mood
###


# Re-run to include labeled emotions and occasional real input events
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# === Emotion PAD directions with labels (from PCMD Table I) ===
emotion_labels = [
    "Hope", "Gratitude", "Admiration", "Gratification", "HappyFor", "Joy", "Love",
    "Pride", "Relief", "Satisfaction", "Gloating", "Remorse", "Disappointment",
    "Fear", "Shame", "Resentment", "Fears-confirmed", "Pity", "Distress",
    "Anger", "Hate", "Reproach"
]

D = np.array([
    [0.2,  0.2,  -0.1], [0.4,  0.2,  -0.3], [0.5,  0.3,  -0.2], [0.6,  0.5,   0.4],
    [0.4,  0.2,   0.2], [0.4,  0.2,   0.1], [0.3,  0.1,   0.2], [0.4,  0.3,   0.3],
    [0.2, -0.3,   0.4], [0.3, -0.2,   0.4], [0.3, -0.3,  -0.1], [-0.3,  0.1, -0.6],
    [-0.3,  0.1, -0.4], [-0.64, 0.60,-0.43], [-0.3,  0.1, -0.6], [-0.2, -0.3, -0.2],
    [-0.5, -0.3, -0.7], [-0.4, -0.2, -0.5], [-0.4, -0.2, -0.5], [-0.51, 0.59,  0.25],
    [-0.6,  0.6,   0.3], [-0.3, -0.1,  0.4]
])
S = len(emotion_labels)
p = np.ones(S) / S

# Big Five to PAD
Q = np.array([
    [0.00,  0.00,  0.21,  0.59,  0.19],
    [0.15,  0.00,  0.00,  0.30, -0.57],
    [0.25,  0.17,  0.60, -0.32,  0.00]
])

F = np.array([0.9, 0.9, 0.3, 0.5, 0.4])
P = Q @ F

# Optimization of phi
from scipy.optimize import minimize

alpha = 2.0
mu_P = 0.1
lambda_e = 1.0
lambda_m = 0.001
event_rate = 1/3.0
a = alpha / lambda_e * (1 - np.exp(-lambda_e / event_rate))

def pcmd_objective(phi):
    U = sum(phi[i] * p[i] * np.outer(D[i], D[i]) for i in range(S))
    y = 0.5 * a * sum(phi[i] * p[i] * D[i] for i in range(S)) + mu_P * a * U @ P
    return np.linalg.norm(y - P)**2 + 0.01 * np.linalg.norm(U, 'fro')**2

res = minimize(pcmd_objective, np.ones(S), bounds=[(0.1, None)] * S)
phi = res.x

print(phi)

# Mood simulation with labeled virtual + occasional real events
M = np.zeros(3)

dt = 0.1
T = 2000.0
steps = int(T / dt)
mood_trace = []
cos_sim_trace = []  # Add array to store cosine similarity values
event_labels = []
closest_emotion_indices = []  # Track which emotion is closest to mood based on cosine
closest_emotion_l2_indices = []  # Track which emotion is closest to mood based on L2

np.random.seed(42)

for step in range(steps):
    t = step * dt

    # Real event: occasionally inject strong real emotions (every N seconds)
    if step % int(10 / dt) == 0 and t > 500 and t < 600:
        idx = emotion_labels.index("Anger")
        I = 5.0  # strong real event
        d = D[idx]
        M += alpha * phi[idx] * I * d * dt
        event_labels.append((t, emotion_labels[idx]))
    elif step % int(10 / dt) == 0 and t > 1000 and t < 1100:
        idx = emotion_labels.index("Relief")
        I = 5.0  # strong real event
        d = D[idx]
        M += alpha * phi[idx] * I * d * dt
        event_labels.append((t, emotion_labels[idx]))
    else:
    # Virtual background event
        idx = np.random.choice(S)
        I = 0.01
        d = D[idx]
        M += alpha * phi[idx] * I * d * dt

    M *= np.exp(-lambda_m * dt)
    
    # Normalize if needed
    if np.max(np.abs(M)) > 1.0:
        M = M / np.max(np.abs(M))
    mood_trace.append(M.copy())
    
    # Calculate cosine similarity between M and P
    # 1 - cosine distance gives cosine similarity
    cos_sim = 1 - cosine(M, P)
    cos_sim_trace.append(cos_sim)
    
    # Find emotion with closest cosine angle to current mood
    emotion_sims = [1 - cosine(M, d) for d in D]
    closest_idx = np.argmax(emotion_sims)
    closest_emotion_indices.append(closest_idx)
    
    # Find emotion with closest L2 distance to current mood
    emotion_l2_dists = [np.linalg.norm(M - d) for d in D]
    closest_l2_idx = np.argmin(emotion_l2_dists)
    closest_emotion_l2_indices.append(closest_l2_idx)

mood_trace = np.array(mood_trace)
cos_sim_trace = np.array(cos_sim_trace)
closest_emotion_indices = np.array(closest_emotion_indices)
closest_emotion_l2_indices = np.array(closest_emotion_l2_indices)

# # Plot

# Create 2D projection plots with emotion labels on PA, PD, and AD
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
axes = axes.flatten()

# Extract event positions and labels
event_indices = [int(t_evt / dt) for t_evt, _ in event_labels]
event_positions = mood_trace[event_indices]
event_texts = [label for _, label in event_labels]

# Pleasure-Arousal
axes[0].plot(mood_trace[:,0], mood_trace[:,1], color='green', alpha=0.3)
for (x, y), label in zip(event_positions[:, [0,1]], event_texts):
    axes[0].text(x, y, label, fontsize=8, color='blue')
for (x, y), label in zip(D[:, [0,1]], emotion_labels):
    axes[0].scatter(x,y, color='black', marker='.', s=20)
    axes[0].text(x, y, label, fontsize=8, color='red')
axes[0].scatter(P[0], P[1], color='black', marker='*', s=80, label='Personality')
axes[0].set_xlabel('Pleasure')
axes[0].set_ylabel('Arousal')
axes[0].set_title('Pleasure vs Arousal')
axes[0].legend()
axes[0].grid(True)
axes[0].set_xlim([-1,1]); axes[0].set_ylim([-1,1])

# Pleasure-Dominance
axes[1].plot(mood_trace[:,0], mood_trace[:,2], color='blue', alpha=0.3)
for (x, y), label in zip(event_positions[:, [0,2]], event_texts):
    axes[1].text(x, y, label, fontsize=8, color='blue')
for (x, y), label in zip(D[:, [0,2]], emotion_labels):
    axes[1].scatter(x,y, color='black', marker='.', s=20)
    axes[1].text(x, y, label, fontsize=8, color='red')
axes[1].scatter(P[0], P[2], color='black', marker='*', s=80, label='Personality')
axes[1].set_xlabel('Pleasure')
axes[1].set_ylabel('Dominance')
axes[1].set_title('Pleasure vs Dominance')
axes[1].legend()
axes[1].grid(True)
axes[1].set_xlim([-1,1]); axes[1].set_ylim([-1,1])

# Arousal-Dominance
axes[2].plot(mood_trace[:,1], mood_trace[:,2], color='purple', alpha=0.3)
for (x, y), label in zip(event_positions[:, [1,2]], event_texts):
    axes[2].text(x, y, label, fontsize=8, color='blue')
for (x, y), label in zip(D[:, [1,2]], emotion_labels):
    axes[2].scatter(x,y, color='black', marker='.', s=20)
    axes[2].text(x, y, label, fontsize=8, color='red')
axes[2].scatter(P[1], P[2], color='black', marker='*', s=80, label='Personality')
axes[2].set_xlabel('Arousal')
axes[2].set_ylabel('Dominance')
axes[2].set_title('Arousal vs Dominance')
axes[2].legend()
axes[2].grid(True)
axes[2].set_xlim([-1,1]); axes[2].set_ylim([-1,1])

# Cosine Similarity between M and P over time
time_steps = np.arange(steps) * dt
axes[3].plot(time_steps, cos_sim_trace, color='red')
axes[3].set_xlabel('Time')
axes[3].set_ylabel('Cosine Similarity')
axes[3].set_title('Mood-Personality Similarity Over Time')
axes[3].grid(True)
axes[3].set_xlim([0, T]); axes[3].set_ylim([-1, 1])

# Plot closest emotion category over time (cosine)
axes[4].scatter(time_steps, closest_emotion_indices, s=10, c='blue', alpha=0.7)
# Add event markers
for t_evt, label in event_labels:
    axes[4].axvline(x=t_evt, color='red', linestyle='--', alpha=0.3)
    
# Customize y-axis to show emotion labels
plt.sca(axes[4])
plt.yticks(range(len(emotion_labels)), emotion_labels, fontsize=8)
axes[4].set_xlabel('Time')
axes[4].set_title('Closest Emotion to Current Mood (Cosine Similarity)')
axes[4].grid(True)
axes[4].set_xlim([0, T])

# Plot closest emotion category over time (L2 distance)
axes[6].scatter(time_steps, closest_emotion_l2_indices, s=10, c='purple', alpha=0.7)
# Add event markers
for t_evt, label in event_labels:
    axes[6].axvline(x=t_evt, color='red', linestyle='--', alpha=0.3)
    
# Customize y-axis to show emotion labels
plt.sca(axes[6])
plt.yticks(range(len(emotion_labels)), emotion_labels, fontsize=8)
axes[6].set_xlabel('Time')
axes[6].set_title('Closest Emotion to Current Mood (L2 Distance)')
axes[6].grid(True)
axes[6].set_xlim([0, T])

# Calculate emotion frequencies (cosine)
if len(closest_emotion_indices) > 0:
    emotion_counts = np.bincount(closest_emotion_indices, minlength=len(emotion_labels))
    emotion_freqs = emotion_counts / len(closest_emotion_indices)
    
    # Sort emotions by frequency
    sorted_indices = np.argsort(emotion_freqs)[::-1]
    sorted_emotions = [emotion_labels[i] for i in sorted_indices]
    sorted_freqs = emotion_freqs[sorted_indices]
    
    # Plot emotion frequency distribution
    bars = axes[5].barh(range(len(sorted_emotions)), sorted_freqs)
    plt.sca(axes[5])
    plt.yticks(range(len(sorted_emotions)), sorted_emotions, fontsize=8)
    axes[5].set_xlabel('Frequency')
    axes[5].set_title('Emotion Frequency Distribution (Cosine)')
    axes[5].grid(True, axis='x')

# Calculate emotion frequencies (L2)
if len(closest_emotion_l2_indices) > 0:
    emotion_l2_counts = np.bincount(closest_emotion_l2_indices, minlength=len(emotion_labels))
    emotion_l2_freqs = emotion_l2_counts / len(closest_emotion_l2_indices)
    
    # Sort emotions by frequency
    sorted_l2_indices = np.argsort(emotion_l2_freqs)[::-1]
    sorted_l2_emotions = [emotion_labels[i] for i in sorted_l2_indices]
    sorted_l2_freqs = emotion_l2_freqs[sorted_l2_indices]
    
    # Plot emotion frequency distribution
    bars = axes[7].barh(range(len(sorted_l2_emotions)), sorted_l2_freqs)
    plt.sca(axes[7])
    plt.yticks(range(len(sorted_l2_emotions)), sorted_l2_emotions, fontsize=8)
    axes[7].set_xlabel('Frequency')
    axes[7].set_title('Emotion Frequency Distribution (L2)')
    axes[7].grid(True, axis='x')

# Plot agreement between cosine and L2 measures
agreement = np.mean(closest_emotion_indices == closest_emotion_l2_indices)
disagreement_indices = np.where(closest_emotion_indices != closest_emotion_l2_indices)[0]


# Create agreement plot
axes[8].plot(time_steps, closest_emotion_indices == closest_emotion_l2_indices, 'o', markersize=2, alpha=0.5)
axes[8].set_xlabel('Time')
axes[8].set_ylabel('Agreement')
axes[8].set_title(f'Agreement Between Cosine and L2 Measures: {agreement:.2%}')
axes[8].set_xlim([0, T])
axes[8].set_ylim([-0.1, 1.1])
for t_evt, label in event_labels:
    axes[8].axvline(x=t_evt, color='red', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
