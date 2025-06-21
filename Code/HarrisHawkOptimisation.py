"""
Harris Hawks Optimization (HHO) for Feature Selection and Parameter Tuning
in Medical Image Segmentation (e.g., Polyp Segmentation).

This script demonstrates HHO's ability to handle high-dimensional optimization
problems by simulating the cooperative hunting behavior of Harris' hawks.

Why HHO is Suitable for Medical Image Segmentation:
---------------------------------------------------
- Handles continuous and binary feature selection (via thresholding).
- Balances exploration and exploitation effectively.
- Particularly strong for sparse and complex search spaces like medical feature maps.
- Capable of escaping local optima, which is crucial in medical tasks where high
  sensitivity is needed.

Author: Your Name
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Objective Function (Example)
# -----------------------------
def fitness_function(x):
    """
    Sphere function (minimization): Sum of squares
    You can replace this with a medical feature selection function, e.g., IOU, Dice error, etc.
    """
    return np.sum(x ** 2)


# ------------------------------------
# Harris Hawks Optimization Algorithm
# ------------------------------------
def hho(fitness_func, lb, ub, dim, n_hawks, max_iter):
    """
    Harris Hawks Optimization core implementation.
    
    Parameters:
    - fitness_func: Function to minimize
    - lb, ub: Lower and upper bounds (scalars or vectors)
    - dim: Dimensionality of the problem
    - n_hawks: Number of hawks (agents)
    - max_iter: Number of iterations

    Returns:
    - best_position: Best solution found
    - best_score: Corresponding fitness value
    - convergence_curve: History of best scores over time
    """

    lb = np.array([lb] * dim)
    ub = np.array([ub] * dim)

    # Initialize the positions of hawks
    X = np.random.uniform(lb, ub, (n_hawks, dim))

    # Initialize fitness and best solution
    best_score = float("inf")
    best_position = np.zeros(dim)
    convergence_curve = []

    # Main loop
    for t in range(max_iter):
        for i in range(n_hawks):
            fitness = fitness_func(X[i, :])
            
            if fitness < best_score:
                best_score = fitness
                best_position = X[i, :].copy()

        # Energy decreases over time
        E1 = 2 * (1 - (t / max_iter))

        for i in range(n_hawks):
            E0 = 2 * np.random.rand() - 1  # -1 to 1
            E = E1 * E0

            if abs(E) >= 1:
                # Exploration phase
                q = np.random.rand()
                rand_hawk_index = np.random.randint(n_hawks)
                X_rand = X[rand_hawk_index, :]

                if q < 0.5:
                    # Random tall tree exploration
                    X[i, :] = X_rand - np.random.rand() * abs(X_rand - 2 * np.random.rand() * X[i, :])
                else:
                    # Perch randomly in range of prey
                    X[i, :] = (best_position - X.mean(0)) - np.random.rand() * (ub - lb) * np.random.rand() + lb

            else:
                # Exploitation phase
                r = np.random.rand()
                J = 2 * (1 - np.random.rand())  # jump strength

                if r >= 0.5 and abs(E) < 0.5:
                    # Soft besiege
                    X[i, :] = best_position - E * abs(J * best_position - X[i, :])

                if r >= 0.5 and abs(E) >= 0.5:
                    # Hard besiege
                    X[i, :] = best_position - E * abs(best_position - X[i, :])

                if r < 0.5:
                    # Soft besiege with progressive rapid dives
                    Y = best_position - E * abs(J * best_position - X[i, :])
                    Y = np.clip(Y, lb, ub)
                    F1 = fitness_func(Y)

                    if F1 < fitness:
                        X[i, :] = Y.copy()
                    else:
                        # Levy flight
                        Z = levy_flight(dim) + np.random.rand() * best_position
                        Z = np.clip(Z, lb, ub)
                        F2 = fitness_func(Z)

                        if F2 < fitness:
                            X[i, :] = Z.copy()

        # Bound enforcement
        X = np.clip(X, lb, ub)
        convergence_curve.append(best_score)

        if t % 10 == 0:
            print(f"[Iteration {t}] Best Score = {best_score:.5f}")

    return best_position, best_score, convergence_curve


def levy_flight(dim):
    """
    Generates Levy flight using Mantegna's algorithm (heavy-tailed distribution)
    """
    beta = 1.5
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
            (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / (np.abs(v) ** (1 / beta))


# ----------------------
# Visualization Utility
# ----------------------
def plot_convergence(curve):
    plt.figure(figsize=(10, 5))
    plt.plot(curve, linewidth=2)
    plt.title("HHO Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------
# Example Usage
# ----------------------
if __name__ == '__main__':
    dim = 30
    lb = -100
    ub = 100
    hawks = 20
    max_iter = 200

    best_pos, best_score, convergence = hho(fitness_function, lb, ub, dim, hawks, max_iter)

    print("\nBest Solution:", best_pos)
    print("Best Fitness:", best_score)
    plot_convergence(convergence)
