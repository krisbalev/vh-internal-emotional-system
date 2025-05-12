# File: plotting.py
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cosine
from config import D, emotion_labels, P, dt

def live_plot_3d(running_flag, shared_history):
    plt.ion()  # interactive mode on
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot static markers for emotion PAD directions
    for x, y, z, label in zip(D[:, 0], D[:, 1], D[:, 2], emotion_labels):
        ax.scatter(x, y, z, color='black', marker='.', s=20)
        ax.text(x, y, z, label, fontsize=8, color='red')
    # Mark personality point
    ax.scatter(P[0], P[1], P[2], color='blue', marker='*', s=80, label='Personality')
    ax.set_xlabel('Pleasure')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    ax.set_title('Mood Trajectory (3D Space)')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.legend()
    ax.grid(True)
    
    # Initialize line and current mood marker
    trajectory_line, = ax.plot([], [], [], color='green', alpha=0.5, label='Trajectory')
    current_mood_marker = ax.scatter([], [], [], color='blue', marker='o', s=50)
    
    while running_flag.value:
        # Read a copy of the shared mood history
        if len(shared_history) > 0:
            times, moods = zip(*list(shared_history))
            moods = np.array(moods)
        else:
            moods = np.empty((0, 3))
        if moods.size > 0:
            trajectory_line.set_data(moods[:, 0], moods[:, 1])
            trajectory_line.set_3d_properties(moods[:, 2])
            current_mood_marker._offsets3d = ([moods[-1, 0]], [moods[-1, 1]], [moods[-1, 2]])
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.5)
    
    plt.ioff()
    plt.show()


def update_2d_plots(running_flag, shared_history):
    plt.ion()  # interactive mode on
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    while running_flag.value:
        if len(shared_history) > 0:
            times, moods = zip(*list(shared_history))
            times = np.array(times)
            mood_trace = np.array(moods)
        else:
            times = np.array([])
            mood_trace = np.empty((0, 3))
        if mood_trace.shape[0] > 0:
            # Compute cosine similarity to personality for each mood sample
            cos_sim_trace = np.array([1 - cosine(m, P) for m in mood_trace])
            closest_emotion_indices = np.array([
                np.argmax([1 - cosine(m, d) for d in D]) for m in mood_trace
            ])
            closest_emotion_l2_indices = np.array([
                np.argmin([np.linalg.norm(m - d) for d in D]) for m in mood_trace
            ])
            emotion_counts = np.bincount(closest_emotion_indices, minlength=len(emotion_labels))
            emotion_freqs = emotion_counts / len(closest_emotion_indices)
            emotion_l2_counts = np.bincount(closest_emotion_l2_indices, minlength=len(emotion_labels))
            emotion_l2_freqs = emotion_l2_counts / len(closest_emotion_l2_indices)
            agreement = np.mean(closest_emotion_indices == closest_emotion_l2_indices)
        else:
            cos_sim_trace = np.array([])
            closest_emotion_indices = np.array([])
            closest_emotion_l2_indices = np.array([])
            emotion_freqs = np.zeros(len(emotion_labels))
            emotion_l2_freqs = np.zeros(len(emotion_labels))
            agreement = 0

        # Subplot 0: Pleasure vs Arousal
        axes[0].cla()
        if mood_trace.shape[0] > 0:
            axes[0].plot(mood_trace[:, 0], mood_trace[:, 1], color='green', alpha=0.3)
        for x, y, label in zip(D[:, 0], D[:, 1], emotion_labels):
            axes[0].scatter(x, y, color='black', marker='.', s=20)
            axes[0].text(x, y, label, fontsize=8, color='red')
        axes[0].scatter(P[0], P[1], color='black', marker='*', s=80, label='Personality')
        axes[0].set_xlabel('Pleasure')
        axes[0].set_ylabel('Arousal')
        axes[0].set_title('Pleasure vs Arousal')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_xlim([-1, 1]); axes[0].set_ylim([-1, 1])

        # Subplot 1: Pleasure vs Dominance
        axes[1].cla()
        if mood_trace.shape[0] > 0:
            axes[1].plot(mood_trace[:, 0], mood_trace[:, 2], color='blue', alpha=0.3)
        for x, z, label in zip(D[:, 0], D[:, 2], emotion_labels):
            axes[1].scatter(x, z, color='black', marker='.', s=20)
            axes[1].text(x, z, label, fontsize=8, color='red')
        axes[1].scatter(P[0], P[2], color='black', marker='*', s=80, label='Personality')
        axes[1].set_xlabel('Pleasure')
        axes[1].set_ylabel('Dominance')
        axes[1].set_title('Pleasure vs Dominance')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_xlim([-1, 1]); axes[1].set_ylim([-1, 1])

        # Subplot 2: Arousal vs Dominance
        axes[2].cla()
        if mood_trace.shape[0] > 0:
            axes[2].plot(mood_trace[:, 1], mood_trace[:, 2], color='purple', alpha=0.3)
        for y, z, label in zip(D[:, 1], D[:, 2], emotion_labels):
            axes[2].scatter(y, z, color='black', marker='.', s=20)
            axes[2].text(y, z, label, fontsize=8, color='red')
        axes[2].scatter(P[1], P[2], color='black', marker='*', s=80, label='Personality')
        axes[2].set_xlabel('Arousal')
        axes[2].set_ylabel('Dominance')
        axes[2].set_title('Arousal vs Dominance')
        axes[2].legend()
        axes[2].grid(True)
        axes[2].set_xlim([-1, 1]); axes[2].set_ylim([-1, 1])

        # Subplot 3: Cosine Similarity over Time
        axes[3].cla()
        if len(times) > 0:
            axes[3].plot(times, cos_sim_trace, color='red')
            axes[3].set_xlim([0, times[-1]])
        axes[3].set_xlabel('Time')
        axes[3].set_ylabel('Cosine Similarity')
        axes[3].set_title('Mood-Personality Similarity Over Time')
        axes[3].grid(True)
        axes[3].set_ylim([-1, 1])

        # Subplot 4: Closest Emotion (Cosine) over Time
        axes[4].cla()
        if len(times) > 0:
            axes[4].scatter(times, closest_emotion_indices, s=10, c='blue', alpha=0.7)
            axes[4].set_xlim([0, times[-1]])
        axes[4].set_xlabel('Time')
        axes[4].set_ylabel('Emotion Index')
        axes[4].set_title('Closest Emotion (Cosine)')
        axes[4].grid(True)
        axes[4].set_yticks(range(len(emotion_labels)))
        axes[4].set_yticklabels(emotion_labels, fontsize=8)

        # Subplot 5: Emotion Frequency Distribution (Cosine)
        axes[5].cla()
        axes[5].barh(range(len(emotion_labels)), emotion_freqs)
        axes[5].set_xlabel('Frequency')
        axes[5].set_title('Emotion Frequency (Cosine)')
        axes[5].grid(True, axis='x')
        axes[5].set_yticks(range(len(emotion_labels)))
        axes[5].set_yticklabels(emotion_labels, fontsize=8)

        # Subplot 6: Closest Emotion (L2) over Time
        axes[6].cla()
        if len(times) > 0:
            axes[6].scatter(times, closest_emotion_l2_indices, s=10, c='purple', alpha=0.7)
            axes[6].set_xlim([0, times[-1]])
        axes[6].set_xlabel('Time')
        axes[6].set_ylabel('Emotion Index')
        axes[6].set_title('Closest Emotion (L2)')
        axes[6].grid(True)
        axes[6].set_yticks(range(len(emotion_labels)))
        axes[6].set_yticklabels(emotion_labels, fontsize=8)

        # Subplot 7: Emotion Frequency Distribution (L2)
        axes[7].cla()
        axes[7].barh(range(len(emotion_labels)), emotion_l2_freqs)
        axes[7].set_xlabel('Frequency')
        axes[7].set_title('Emotion Frequency (L2)')
        axes[7].grid(True, axis='x')
        axes[7].set_yticks(range(len(emotion_labels)))
        axes[7].set_yticklabels(emotion_labels, fontsize=8)

        # Subplot 8: Agreement Between Cosine and L2 Measures
        axes[8].cla()
        if len(times) > 0:
            agreement_arr = (closest_emotion_indices == closest_emotion_l2_indices).astype(int)
            axes[8].plot(times, agreement_arr, 'o', markersize=2, alpha=0.5)
            axes[8].set_xlim([0, times[-1]])
        axes[8].set_xlabel('Time')
        axes[8].set_ylabel('Agreement')
        axes[8].set_title(f'Agreement Between Cosine and L2: {agreement:.2%}')
        axes[8].grid(True)
        axes[8].set_ylim([-0.1, 1.1])
        
        fig.tight_layout()
        fig.canvas.draw_idle()
        plt.pause(0.5)
    
    plt.ioff()
    plt.show()
