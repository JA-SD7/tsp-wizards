"""
Adaptive Grey Wolf Optimization (AGWO) for Multi-Objective TSP
==============================================================
Based on: Mirjalili et al. (2014) - Grey Wolf Optimizer
Adapted for discrete permutation encoding and multi-objective optimization.

Usage:
    python agwo_tsp.py

Author: [Your Name]
Course: DSAI3203
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
from copy import deepcopy


# =============================================================================
# CITY / DATASET CONFIGURATION
# Place eil51.tsp or berlin52.tsp in the same folder as this script.
# =============================================================================
def load_tsplib(filename):
    """
    Load city coordinates from a TSPLIB .tsp file.

    Reads the NODE_COORD_SECTION block and extracts (x, y) coordinates.
    Compatible with EUC_2D format files like eil51.tsp and berlin52.tsp.

    Args:
        filename (str): Path to the .tsp file

    Returns:
        np.ndarray: Array of shape (n_cities, 2) with x,y coordinates
    """
    coords = []
    reading = False
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == 'NODE_COORD_SECTION':
                reading = True
                continue
            if line == 'EOF':
                break
            if reading:
                parts = line.split()
                # Each line: node_index x y
                coords.append((float(parts[1]), float(parts[2])))
    return np.array(coords)


def compute_distance_matrix(coords):
    """
    Compute the Euclidean distance matrix between all city pairs.

    Args:
        coords (np.ndarray): City coordinates (n x 2)

    Returns:
        np.ndarray: Distance matrix of shape (n x n)
    """
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = np.sqrt(
                    (coords[i][0] - coords[j][0])**2 +
                    (coords[i][1] - coords[j][1])**2
                )
    return dist


# =============================================================================
# [MODIFY ZONE 2] - OBJECTIVE FUNCTIONS
# Currently: f1 = total distance, f2 = total time (distance / speed).
# =============================================================================
def compute_time_matrix(dist_matrix, seed=99):
    """
    Generate a travel time matrix by dividing distances by random speeds.
    Speed varies per edge to simulate real road conditions.

    Args:
        dist_matrix (np.ndarray): Distance matrix
        seed (int): Random seed

    Returns:
        np.ndarray: Time matrix of shape (n x n)
    """
    np.random.seed(seed)
    n = len(dist_matrix)
    speeds = np.random.uniform(40, 120, (n, n))
    speeds[speeds == 0] = 1
    time_matrix = dist_matrix / speeds
    return time_matrix


def tour_distance(tour, dist_matrix):
    """Calculate total distance of a tour (f1 objective)."""
    total = 0
    n = len(tour)
    for i in range(n):
        total += dist_matrix[tour[i]][tour[(i + 1) % n]]
    return total


def tour_time(tour, time_matrix):
    """Calculate total travel time of a tour (f2 objective)."""
    total = 0
    n = len(tour)
    for i in range(n):
        total += time_matrix[tour[i]][tour[(i + 1) % n]]
    return total


def evaluate(tour, dist_matrix, time_matrix):
    """
    Evaluate a tour on both objectives.

    Returns:
        tuple: (f1_distance, f2_time)
    """
    f1 = tour_distance(tour, dist_matrix)
    f2 = tour_time(tour, time_matrix)
    return f1, f2


# =============================================================================
# [MODIFY ZONE 3] - SWAP OPERATOR (core discrete move)
# =============================================================================
def swap_operator(tour, n_swaps=1):
    """
    Apply n random city swaps to a tour to simulate a 'leap' in discrete space.

    Args:
        tour (list): Current tour (permutation of city indices)
        n_swaps (int): Number of swaps to apply

    Returns:
        list: New tour after applying swaps
    """
    new_tour = tour[:]
    for _ in range(n_swaps):
        i, j = random.sample(range(len(new_tour)), 2)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def two_opt_swap(tour):
    """
    [MODIFY ZONE 3 - ALTERNATIVE] 2-opt improvement move.
    Reverses a random sub-segment of the tour.
    """
    new_tour = tour[:]
    i, j = sorted(random.sample(range(len(tour)), 2))
    new_tour[i:j+1] = reversed(new_tour[i:j+1])
    return new_tour


# =============================================================================
# PARETO ARCHIVE - Multi-objective solution management
# =============================================================================
def dominates(sol_a, sol_b):
    """
    Check if solution a dominates solution b (Pareto dominance).

    Args:
        sol_a, sol_b: tuples of (f1, f2) objective values

    Returns:
        bool: True if a dominates b
    """
    return (sol_a[0] <= sol_b[0] and sol_a[1] <= sol_b[1] and
            (sol_a[0] < sol_b[0] or sol_a[1] < sol_b[1]))


def update_pareto_archive(archive, new_tour, new_obj):
    """
    Update the Pareto archive with a new solution.

    Args:
        archive (list): List of (tour, objectives) tuples
        new_tour (list): New candidate tour
        new_obj (tuple): Objectives (f1, f2) of new tour

    Returns:
        list: Updated archive
    """
    for _, obj in archive:
        if dominates(obj, new_obj):
            return archive
    archive = [(t, o) for t, o in archive if not dominates(new_obj, o)]
    archive.append((new_tour[:], new_obj))
    return archive


# =============================================================================
# [MODIFY ZONE 4] - AGWO LEADER SELECTION STRATEGY
# =============================================================================
def select_leaders(archive):
    """
    Select alpha, beta, delta leaders from the Pareto archive.

    Args:
        archive (list): Current Pareto archive

    Returns:
        tuple: (alpha_tour, beta_tour, delta_tour)
    """
    if len(archive) >= 3:
        leaders = random.sample(archive, 3)
    elif len(archive) == 2:
        leaders = archive + [archive[0]]
    else:
        leaders = archive * 3

    return leaders[0][0], leaders[1][0], leaders[2][0]


# =============================================================================
# [MODIFY ZONE 5] - AGWO POSITION UPDATE
# =============================================================================
def move_toward_leader(omega_tour, leader_tour, a, n_cities):
    """
    Move an omega wolf toward a leader by applying swap operations.

    Args:
        omega_tour (list): Current wolf position (tour)
        leader_tour (list): Leader's tour to move toward
        a (float): Adaptive parameter (decreases 2 -> 0 over iterations)
        n_cities (int): Number of cities

    Returns:
        list: Updated tour after moving toward leader
    """
    n_swaps = max(1, int(a * n_cities / 6))
    new_tour = swap_operator(omega_tour, n_swaps)
    return new_tour


# =============================================================================
# MAIN AGWO ALGORITHM
# =============================================================================
def agwo_motsp(n_cities, dist_matrix, time_matrix,
               pop_size=30, max_iter=300, archive_max=100):
    """
    Adaptive Grey Wolf Optimization for Multi-Objective TSP.

    Args:
        n_cities (int): Number of cities
        dist_matrix (np.ndarray): Distance cost matrix
        time_matrix (np.ndarray): Time cost matrix
        pop_size (int): Number of wolves in the population
        max_iter (int): Maximum number of iterations
        archive_max (int): Maximum Pareto archive size

    Returns:
        tuple: (pareto_archive, convergence_f1, convergence_f2)
    """
    population = [random.sample(range(n_cities), n_cities)
                  for _ in range(pop_size)]

    archive = []
    for wolf in population:
        obj = evaluate(wolf, dist_matrix, time_matrix)
        archive = update_pareto_archive(archive, wolf, obj)

    convergence_f1 = []
    convergence_f2 = []

    for iteration in range(max_iter):

        a = 2 * (1 - (iteration / max_iter) ** 2)

        alpha, beta, delta = select_leaders(archive)

        new_population = []
        for wolf in population:
            x1 = move_toward_leader(wolf, alpha, a, n_cities)
            x2 = move_toward_leader(wolf, beta, a, n_cities)
            x3 = move_toward_leader(wolf, delta, a, n_cities)

            candidates = [x1, x2, x3, wolf]
            best_candidate = min(
                candidates,
                key=lambda t: (
                    evaluate(t, dist_matrix, time_matrix)[0] +
                    evaluate(t, dist_matrix, time_matrix)[1]
                )
            )
            new_population.append(best_candidate)

            obj = evaluate(best_candidate, dist_matrix, time_matrix)
            archive = update_pareto_archive(archive, best_candidate, obj)

        population = new_population

        if len(archive) > archive_max:
            archive = archive[:archive_max]

        best_f1 = min(o[0] for _, o in archive)
        best_f2 = min(o[1] for _, o in archive)
        convergence_f1.append(best_f1)
        convergence_f2.append(best_f2)

        if (iteration + 1) % 50 == 0:
            print(f"  AGWO Iter {iteration+1}/{max_iter} | "
                  f"Archive size: {len(archive)} | "
                  f"Best f1: {best_f1:.2f} | Best f2: {best_f2:.4f}")

    return archive, convergence_f1, convergence_f2


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_pareto_front(archive, title="AGWO Pareto Front"):
    """Plot the Pareto front in objective space."""
    f1_vals = [o[0] for _, o in archive]
    f2_vals = [o[1] for _, o in archive]
    plt.figure(figsize=(7, 5))
    plt.scatter(f1_vals, f2_vals, c='steelblue', s=60, edgecolors='navy', alpha=0.7)
    plt.xlabel("f1: Total Distance")
    plt.ylabel("f2: Total Travel Time")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("agwo_pareto_front.png", dpi=150)
    plt.close()
    print("  Saved: agwo_pareto_front.png")


def plot_convergence(conv_f1, conv_f2, label="AGWO"):
    """Plot convergence curves for both objectives."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(conv_f1, color='steelblue')
    ax1.set_title(f"{label} Convergence - f1 (Distance)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Best Distance")
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(conv_f2, color='darkorange')
    ax2.set_title(f"{label} Convergence - f2 (Time)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best Time")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{label.lower()}_convergence.png", dpi=150)
    plt.close()
    print(f"  Saved: {label.lower()}_convergence.png")


def plot_best_tour(tour, coords, title="Best Tour"):
    """Plot the best tour on a city map."""
    tour_coords = [coords[i] for i in tour] + [coords[tour[0]]]
    xs = [c[0] for c in tour_coords]
    ys = [c[1] for c in tour_coords]
    plt.figure(figsize=(7, 7))
    plt.plot(xs, ys, 'b-o', markersize=8)
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=5)
    for i, (x, y) in enumerate(coords):
        plt.annotate(str(i), (x, y), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("agwo_best_tour.png", dpi=150)
    plt.close()
    print("  Saved: agwo_best_tour.png")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  AGWO for Multi-Objective TSP")
    print("=" * 60)

    # ==========================================================================
    # [MODIFY] Switch between datasets by changing the filename below.
    # Options: 'eil51.tsp' (51 cities) or 'berlin52.tsp' (52 cities)
    # Make sure the .tsp file is in the same folder as this script.
    # ==========================================================================
    TSP_FILE = 'eil51.tsp'
    POP_SIZE = 30
    MAX_ITER = 300
    ARCHIVE_MAX = 100

    coords = load_tsplib(TSP_FILE)
    N_CITIES = len(coords)
    dist_matrix = compute_distance_matrix(coords)
    time_matrix = compute_time_matrix(dist_matrix, seed=99)

    print(f"\nLoaded: {TSP_FILE} | Cities: {N_CITIES}")
    print(f"Setup: Pop: {POP_SIZE} | Iter: {MAX_ITER}")
    print("\nRunning AGWO...")
    start = time.time()

    archive, conv_f1, conv_f2 = agwo_motsp(
        N_CITIES, dist_matrix, time_matrix,
        pop_size=POP_SIZE,
        max_iter=MAX_ITER,
        archive_max=ARCHIVE_MAX
    )

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"Pareto archive size: {len(archive)}")
    print(f"Best f1 (distance): {min(o[0] for _, o in archive):.2f}")
    print(f"Best f2 (time):     {min(o[1] for _, o in archive):.4f}")

    plot_pareto_front(archive)
    plot_convergence(conv_f1, conv_f2, label="AGWO")
    best_tour = min(archive, key=lambda x: x[1][0])[0]
    plot_best_tour(best_tour, coords, title="AGWO Best Tour (min distance)")
    print("\nDone.")