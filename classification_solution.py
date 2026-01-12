#!/usr/bin/env python3
"""
Question 2 Solution (SVM + Genetic Algorithm) — Linear classification with outliers.

- Loads CSV with columns: No, x, y, Label
- Saves figures into ./outputs/
  fig1_scatter.png
  fig2_svm_boundary.png
  fig3_ga_boundary.png
- Trains:
  (1) Linear soft-margin SVM (kernel='linear') with small CV grid search for C
  (2) GA-optimized linear decision boundary w1*x + w2*y + b = 0
- Prints metrics: Accuracy, F1, Confusion Matrix (for each method)

Dependencies: numpy, pandas, matplotlib, scikit-learn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
rng = np.random.default_rng(SEED)


# -----------------------------
# Utility plotting
# -----------------------------
def scatter_base(ax, X, y):
    X0 = X[y == 0]
    X1 = X[y == 1]
    ax.scatter(X0[:, 0], X0[:, 1], label="Class 0 (standard)", marker="o")
    ax.scatter(X1[:, 0], X1[:, 1], label="Class 1 (anomaly)", marker="^")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)


def plot_boundary(ax, w1, w2, b):
    xmin, xmax = ax.get_xlim()
    xs = np.linspace(xmin, xmax, 200)
    if abs(w2) < 1e-10:
        if abs(w1) > 1e-10:
            x0 = -b / w1
            ax.axvline(x0, linestyle="--", label="Decision boundary")
    else:
        ys = -(w1 * xs + b) / w2
        ax.plot(xs, ys, linestyle="--", label="Decision boundary")


# -----------------------------
# GA components
# -----------------------------
def _fitness(ind, X, y):
    """
    Fitness = F1 + 0.1 * sigmoid(normalized_margin) - 0.001 * L2_penalty

    This encourages:
    - high F1 (robust to outliers)
    - larger correct-class margin (but normalized to avoid weight explosion)
    - small weights (regularization)
    """
    w1, w2, b = ind
    scores = X[:, 0] * w1 + X[:, 1] * w2 + b
    preds = (scores >= 0).astype(int)

    f1 = f1_score(y, preds, zero_division=0)

    t = 2 * y - 1  # {0,1} -> {-1,+1}
    norm_w = np.linalg.norm([w1, w2]) + 1e-8
    margin = float(np.mean(t * scores) / norm_w)
    margin_reward = 1.0 / (1.0 + np.exp(-margin))

    penalty = 0.001 * (w1 * w1 + w2 * w2 + b * b)
    return f1 + 0.1 * margin_reward - penalty


def _tournament_select(pop, fits, k=3):
    idx = rng.integers(0, len(pop), size=k)
    best = idx[np.argmax(fits[idx])]
    return pop[best].copy()


def _crossover(p1, p2):
    a = rng.random()
    c1 = a * p1 + (1 - a) * p2
    c2 = (1 - a) * p1 + a * p2
    return c1, c2


def _mutate(ind, sigma=0.5, p=0.2):
    for i in range(len(ind)):
        if rng.random() < p:
            ind[i] += rng.normal(0, sigma)
    return ind


def run_ga(X, y, pop_size=200, generations=300, elite=2, stall=60):
    pop = rng.normal(0, 2, size=(pop_size, 3))
    best_fit = -1e18
    best_ind = None
    stall_count = 0

    for gen in range(generations):
        fits = np.array([_fitness(ind, X, y) for ind in pop])
        order = np.argsort(fits)[::-1]
        pop = pop[order]
        fits = fits[order]

        if fits[0] > best_fit + 1e-8:
            best_fit = float(fits[0])
            best_ind = pop[0].copy()
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= stall:
            break

        new_pop = []
        new_pop.extend(pop[:elite])  # elitism

        while len(new_pop) < pop_size:
            p1 = _tournament_select(pop, fits)
            p2 = _tournament_select(pop, fits)
            c1, c2 = _crossover(p1, p2)
            c1 = _mutate(c1)
            c2 = _mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = np.array(new_pop)

    return best_ind, best_fit, gen + 1


# -----------------------------
# Main
# -----------------------------
def main():
    csv_path = "DataKlasifikasi(Sheet1).csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. Put the CSV in the same folder as this script "
            "or edit csv_path."
        )

    df = pd.read_csv(csv_path)
    X = df[["x", "y"]].to_numpy(dtype=float)
    y = df["Label"].to_numpy(dtype=int)

    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)

    # (1) Scatter plot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    scatter_base(ax, X, y)
    ax.set_title("Dataset Scatter Plot (x vs y) colored by label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_scatter.png"))
    plt.close(fig)

    # -------------------------
    # (2) Method 1: Linear SVM
    # -------------------------
    param_grid = {"C": [0.1, 1, 10, 100]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    svm = SVC(kernel="linear", random_state=SEED)

    gs = GridSearchCV(svm, param_grid, scoring="f1", cv=cv)
    gs.fit(X, y)
    best_C = gs.best_params_["C"]

    svm_best = gs.best_estimator_
    svm_best.fit(X, y)

    pred_svm = svm_best.predict(X)
    acc_svm = accuracy_score(y, pred_svm)
    f1_svm = f1_score(y, pred_svm)
    cm_svm = confusion_matrix(y, pred_svm)

    # decision boundary parameters: w1*x + w2*y + b = 0
    w = svm_best.coef_[0]
    b = float(svm_best.intercept_[0])

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    scatter_base(ax, X, y)
    plot_boundary(ax, float(w[0]), float(w[1]), b)
    ax.set_title(f"Linear SVM Decision Boundary (C={best_C})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_svm_boundary.png"))
    plt.close(fig)

    # -------------------------
    # (3) Method 2: GA Classifier
    # -------------------------
    best_ind, best_fit, used_gens = run_ga(X, y)
    w1, w2, b0 = [float(v) for v in best_ind]

    scores = X[:, 0] * w1 + X[:, 1] * w2 + b0
    pred_ga = (scores >= 0).astype(int)

    acc_ga = accuracy_score(y, pred_ga)
    f1_ga = f1_score(y, pred_ga)
    cm_ga = confusion_matrix(y, pred_ga)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    scatter_base(ax, X, y)
    plot_boundary(ax, w1, w2, b0)
    ax.set_title("GA-Optimized Linear Decision Boundary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_ga_boundary.png"))
    plt.close(fig)

    # -------------------------
    # Print results
    # -------------------------
    print("===== RESULTS (evaluated on the full dataset) =====")
    print("\n--- Linear SVM ---")
    print(f"Chosen C (5-fold CV, maximize F1): {best_C}")
    print(f"Accuracy: {acc_svm:.4f}")
    print(f"F1-score: {f1_svm:.4f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:")
    print(cm_svm)

    print("\n--- Genetic Algorithm (GA) ---")
    print(f"GA best fitness: {best_fit:.6f} (gens used: {used_gens})")
    print(f"Best parameters [w1,w2,b]: [{w1:.6f}, {w2:.6f}, {b0:.6f}]")
    print(f"Accuracy: {acc_ga:.4f}")
    print(f"F1-score: {f1_ga:.4f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:")
    print(cm_ga)

    print("\nFigures saved in:", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
