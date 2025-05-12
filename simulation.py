import time
import threading
import numpy as np
from config import P, dt, lambda_m, D, S, alpha
from optimizer import phi

# Global variables for simulation
global_M = P.copy()  # starting mood at personality point
mood_history = []
start_time = time.time()
sim_lock = threading.Lock()
running_flag = None  # set by main

# Mood update loop
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