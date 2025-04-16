import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.optimize import minimize
import threading, time
from transformers import pipeline
from multiprocessing import Manager, Process

# -----------------------------
# Parameters and Initialization
# -----------------------------

# Define emotion labels and PAD directions
emotion_labels = [
    "Hope", "Gratitude", "Admiration", "Gratification", "HappyFor", "Joy", "Love",
    "Pride", "Relief", "Satisfaction", "Gloating", "Remorse", "Disappointment",
    "Fear", "Shame", "Resentment", "Fears-confirmed", "Pity", "Distress",
    "Anger", "Hate", "Reproach"
]

D = np.array([
    [0.2,  0.2,  -0.1],
    [0.4,  0.2,  -0.3],
    [0.5,  0.3,  -0.2],
    [0.6,  0.5,   0.4],
    [0.4,  0.2,   0.2],
    [0.4,  0.2,   0.1],
    [0.3,  0.1,   0.2],
    [0.4,  0.3,   0.3],
    [0.2, -0.3,   0.4],
    [0.3, -0.2,   0.4],
    [0.3, -0.3,  -0.1],
    [-0.3,  0.1, -0.6],
    [-0.3,  0.1, -0.4],
    [-0.64, 0.60, -0.43],
    [-0.3,  0.1, -0.6],
    [-0.2, -0.3, -0.2],
    [-0.5, -0.3, -0.7],
    [-0.4, -0.2, -0.5],
    [-0.4, -0.2, -0.5],
    [-0.51, 0.59,  0.25],
    [-0.6,  0.6,   0.3],
    [-0.3, -0.1,  0.4]
])
S = len(emotion_labels)
p = np.ones(S) / S

# Big Five to PAD conversion matrices and personality vector
Q = np.array([
    [0.00,  0.00,  0.21,  0.59,  0.19],
    [0.15,  0.00,  0.00,  0.30, -0.57],
    [0.25,  0.17,  0.60, -0.32,  0.00]
])
F = np.array([0.9, 0.9, 0.3, 0.5, 0.4])
P = Q @ F   # Personality point in PAD

# Optimize phi weights (forcing phi > 0.1)
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
print("Optimized phi weights:", phi)

# -----------------------------
# Global Variables for Simulation
# -----------------------------
dt = 0.1  # time step (seconds)
global_M = P.copy()  # starting mood at personality point
# 'mood_history' will be replaced by a shared list from Manager later.
mood_history = []
start_time = time.time()
# Lock for simulation updates in the main process
sim_lock = threading.Lock()

# Initialize text classifier (this may take a few seconds)
print("Loading classifier model ...")
classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None
)
print("Classifier loaded.")

# -----------------------------
# Simulation Update Function
# -----------------------------
def update_mood():
    global global_M
    while running_flag.value:
        with sim_lock:
            # Virtual event: randomly select an emotion index
            idx = np.random.choice(S)
            I = 0.01  # low intensity for background events
            d = D[idx]
            # Update mood: add event influence and apply decay
            global_M += alpha * phi[idx] * I * d * dt
            global_M *= np.exp(-lambda_m * dt)
            # Normalize if necessary
            if np.max(np.abs(global_M)) > 1.0:
                global_M = global_M / np.max(np.abs(global_M))
            # Save current time and mood
            mood_history.append((time.time() - start_time, global_M.copy()))
        time.sleep(dt)

# -----------------------------
# User Input Processing Function
# -----------------------------
def process_user_input():
    global global_M
    while running_flag.value:
        user_text = input("Enter your message (or type 'quit' to exit): ")
        if user_text.lower() in ['quit', 'exit']:
            running_flag.value = False
            break

        try:
            results = classifier(user_text)
            # Unwrap nested results if needed
            if isinstance(results, list) and results and isinstance(results[0], list):
                results = results[0]
            # Pick the prediction with the highest score
            best_pred = max(results, key=lambda x: x['score'])
            predicted_label = best_pred['label']
            print(f"Classifier predicted: {predicted_label} (score: {best_pred['score']:.3f})")
        except Exception as e:
            print("Error during classification:", e)
            predicted_label = None

        # Determine the current emotion based on simulation mood
        with sim_lock:
            emotion_sims = [1 - cosine(global_M, d) for d in D]
            closest_idx = np.argmax(emotion_sims)
            current_emotion = emotion_labels[closest_idx]
        print(f"Current emotion (simulation): {current_emotion}")

        # Map classifier output to one of our emotion labels (ignoring case)
        mapped_label = None
        if predicted_label is not None:
            for label in emotion_labels:
                if label.lower() == predicted_label.lower():
                    mapped_label = label
                    break

        if mapped_label is None:
            print("No direct mapping found for the classifier label; no event applied.")
            continue

        # Apply a real user event if a mapping is found
        idx = emotion_labels.index(mapped_label)
        I = 5.0  # higher intensity for user events
        with sim_lock:
            global_M += alpha * phi[idx] * I * D[idx] * dt
            mood_history.append((time.time() - start_time, global_M.copy()))
        print(f"Applied user event corresponding to emotion: {mapped_label}")

# -----------------------------
# 3D Live Plotting Function (Process)
# -----------------------------
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
    current_mood = ax.scatter([], [], [], color='blue', marker='o', s=50)
    
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
            current_mood._offsets3d = ([moods[-1, 0]], [moods[-1, 1]], [moods[-1, 2]])
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.5)
    
    plt.ioff()
    plt.show()

# -----------------------------
# 2D Live Plotting Function (Process)
# -----------------------------
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

# -----------------------------
# Main Execution: Set Up Threads and Processes
# -----------------------------
if __name__ == '__main__':
    # Use a Manager to create shared objects for the plot processes.
    manager = Manager()
    shared_mood_history = manager.list()
    # Replace the local mood_history with the shared list.
    mood_history = shared_mood_history
    running_flag = manager.Value('b', True)

    # Start the simulation and user input threads in the main process.
    sim_thread = threading.Thread(target=update_mood)
    input_thread = threading.Thread(target=process_user_input)
    sim_thread.start()
    input_thread.start()
    
    # Start the 3D live plot in its own process.
    plot3d_proc = Process(target=live_plot_3d, args=(running_flag, shared_mood_history))
    plot3d_proc.start()
    
    # Start the 2D update plots in a second process.
    plot2d_proc = Process(target=update_2d_plots, args=(running_flag, shared_mood_history))
    plot2d_proc.start()
    
    # Wait for the simulation and user input threads to finish.
    sim_thread.join()
    input_thread.join()
    
    # Signal the plotting processes to stop and wait for them.
    running_flag.value = False
    plot3d_proc.join()
    plot2d_proc.join()
    
    print("Simulation ended.")
