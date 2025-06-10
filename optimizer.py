import numpy as np
from scipy.optimize import minimize
from config import D, p, a, mu_P, P, S

# Objective for optimizing phi weights
def pcmd_objective(phi):
    U = sum(phi[i] * p[i] * np.outer(D[i], D[i]) for i in range(S))
    y = 0.5 * a * sum(phi[i] * p[i] * D[i] for i in range(S)) + mu_P * a * U @ P
    return np.linalg.norm(y - P)**2 + 0.01 * np.linalg.norm(U, 'fro')**2

# Optimize phi with bounds phi > 0.1
res = minimize(pcmd_objective, np.ones(S), bounds=[(0.1, None)] * S)
phi = res.x
# print("Optimized phi weights:", phi)


