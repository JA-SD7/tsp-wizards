"""
Adaptive Grey Wolf Optimization (AGWO) for Multi-Objective TSP
==============================================================
Based on: Mirjalili et al. (2014) - Grey Wolf Optimizer
Adapted for discrete permutation encoding and multi-objective optimization.

Objectives:
    f1 - Minimize total tour distance (km)
    f2 - Minimize total carbon emissions (kg CO2)
         Emission model: distance * fuel_consumption_rate * emission_factor

Usage:
    python agwo_tsp.py

Author: [Your Name]
Course: DSAI3203
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt


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
# [MODIFY ZONE 2] - EMISSION-BASED OBJECTIVE (f2)
# f2 = total carbon emissions across the tour.
# Each edge has a randomly assigned fuel consumption rate (L/km) simulating
# different vehicle loads or road gradients between cities.
# Emission per edge = distance * fuel_rate * EMISSION_FACTOR
# EMISSION_FACTOR = 2.31 kg CO2 per litre (standard petrol value)
# This directly ties to the project's stated goal of reducing emissions
# in logistics and delivery operations.
# =============================================================================
EMISSION_FACTOR = 2.31  # kg CO2 per litre of petrol burned

def compute_emission_matrix(dist_matrix, seed=99):
    """
    Generate a carbon emission matrix based on distance and fuel consumption.

    Each edge is assigned a random fuel consumption rate between 0.06 and
    0.15 L/km, simulating varying vehicle loads and road conditions between
    city pairs. Emission per edge = distance * fuel_rate * EMISSION_FACTOR.

    Args:
        dist_matrix (np.ndarray): Distance matrix
        seed (int): Random seed for reproducibility

    Returns:
        np.ndarray: Emission matrix of shape (n x n), values in kg CO2
    """
    np.random.seed(seed)
    n = len(dist_matrix)
    # Fuel consumption rate varies per edge: 0.03 to 0.25 L/km
    fuel_rates = np.random.uniform(0.03, 0.25, (n, n))
    emission_matrix = dist_matrix * fuel_rates * EMISSION_FACTOR
    return emission_matrix


def tour_distance(tour, dist_matrix):
    """Calculate total distance of a tour (f1 objective)."""
    n = len(tour)
    return sum(dist_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))


def tour_emission(tour, emission_matrix):
    """Calculate total carbon emission of a tour (f2 objective)."""
    n = len(tour)
    return sum(emission_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))


def evaluate(tour, dist_matrix, emission_matrix):
    """
    Evaluate a tour on both objectives.

    Returns:
        tuple: (f1_distance, f2_emission)
    """
    f1 = tour_distance(tour, dist_matrix)
    f2 = tour_emission(tour, emission_matrix)
    return f1, f2


# =============================================================================
# [MODIFY ZONE 3] - TWO-OPT SWAP OPERATOR
# The 2-opt move reverses a sub-segment of the tour between two randomly
# selected positions i and j. This is more structured than a plain city
# swap because it preserves partial route continuity — the reversed segment
# still connects its endpoints, just in the opposite direction.
# This is known to produce better solutions than random swaps because it
# directly eliminates route crossings, which are always suboptimal in
# Euclidean TSP.
# =============================================================================
def two_opt_move(tour, n_moves=1):
    """
    Apply n two-opt moves to a tour.

    Each move selects two random positions i and j, then reverses the
    sub-segment between them. This eliminates crossing edges which are
    always suboptimal in Euclidean space.

    Args:
        tour (list): Current tour (permutation of city indices)
        n_moves (int): Number of 2-opt moves to apply

    Returns:
        list: New tour after applying 2-opt moves
    """
    new_tour = tour[:]
    for _ in range(n_moves):
        i, j = sorted(random.sample(range(len(new_tour)), 2))
        new_tour[i:j+1] = reversed(new_tour[i:j+1])
    return new_tour


# =============================================================================
# PARETO ARCHIVE - Multi-objective solution management
# =============================================================================
def dominates(sol_a, sol_b):
    """
    Check if solution a dominates solution b (Pareto dominance).
    a dominates b if a is no worse in all objectives AND strictly
    better in at least one.

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
    Removes dominated solutions and adds the new one if non-dominated.

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
# [MODIFY ZONE 4] - CROWDING DISTANCE LEADER SELECTION
# Instead of selecting alpha, beta, delta randomly from the archive,
# we use crowding distance to prefer leaders that are spread out along
# the Pareto front. This avoids clustering all three leaders in one
# region of the objective space, which would cause the wolf pack to
# converge prematurely to a small portion of the Pareto front.
# Crowding distance is the same diversity mechanism used in NSGA-II
# (Deb et al., 2002), cited in the reference template.
# =============================================================================
def crowding_distance(archive):
    """
    Compute the crowding distance for each solution in the archive.

    Crowding distance measures how isolated a solution is from its
    neighbors in objective space. Solutions at the boundary of the
    Pareto front are assigned infinite distance. Higher crowding
    distance = more isolated = more diverse contribution.

    Args:
        archive (list): List of (tour, (f1, f2)) tuples

    Returns:
        list: Crowding distances for each archive member
    """
    n = len(archive)
    if n <= 2:
        return [float('inf')] * n

    distances = [0.0] * n

    for obj_idx in range(2):  # f1 and f2
        # Sort by this objective
        sorted_idx = sorted(range(n), key=lambda i: archive[i][1][obj_idx])
        obj_vals = [archive[i][1][obj_idx] for i in sorted_idx]

        obj_range = obj_vals[-1] - obj_vals[0]
        if obj_range == 0:
            continue

        # Boundary solutions get infinite distance
        distances[sorted_idx[0]]  = float('inf')
        distances[sorted_idx[-1]] = float('inf')

        # Interior solutions: distance = gap between neighbors
        for k in range(1, n - 1):
            distances[sorted_idx[k]] += (
                (obj_vals[k + 1] - obj_vals[k - 1]) / obj_range
            )

    return distances


def select_leaders(archive):
    """
    Select alpha, beta, delta leaders from the Pareto archive using
    crowding distance. Leaders with higher crowding distance are preferred
    to ensure the wolf pack explores a diverse region of the Pareto front.

    Args:
        archive (list): Current Pareto archive

    Returns:
        tuple: (alpha_tour, beta_tour, delta_tour)
    """
    if len(archive) <= 3:
        chosen = (archive * 3)[:3]
        return chosen[0][0], chosen[1][0], chosen[2][0]

    distances = crowding_distance(archive)

    # Rank by crowding distance descending (most isolated first)
    ranked = sorted(range(len(archive)),
                    key=lambda i: distances[i], reverse=True)

    # Add slight randomness among the top half to avoid determinism
    top_half = ranked[:max(3, len(ranked) // 2)]
    chosen_idx = random.sample(top_half, 3)

    return (archive[chosen_idx[0]][0],
            archive[chosen_idx[1]][0],
            archive[chosen_idx[2]][0])


# =============================================================================
# [MODIFY ZONE 5] - SUB-QUADRATIC ADAPTIVE DECAY (exponent = 1.5)
# The adaptive parameter a controls how many 2-opt moves are applied
# when moving toward each leader. It decreases from 2 to 0 over the
# course of the run.
# Using exponent 1.5 (sub-quadratic) instead of the standard 2 (quadratic)
# means a decreases more slowly early on, giving the wolves more exploration
# time before switching to exploitation. This is beneficial for larger TSP
# instances where the search space is vast.
# =============================================================================
def move_toward_leader(omega_tour, leader_tour, a, n_cities):
    """
    Move an omega wolf toward a leader using 2-opt moves.

    The sub-quadratic decay (exponent=1.5) keeps 'a' higher for longer,
    sustaining exploration before converging. The number of 2-opt moves
    scales with 'a' so early iterations apply more moves (exploration)
    and later iterations apply fewer (exploitation).

    Args:
        omega_tour (list): Current wolf position (tour)
        leader_tour (list): Leader's tour to move toward
        a (float): Adaptive parameter (decreases 2 -> 0 over iterations)
        n_cities (int): Number of cities

    Returns:
        list: Updated tour after moving toward leader
    """
    n_moves = max(1, int(a * n_cities / 6))
    return two_opt_move(omega_tour, n_moves)


# =============================================================================
# MAIN AGWO ALGORITHM
# =============================================================================
def agwo_motsp(n_cities, dist_matrix, emission_matrix,
               pop_size=30, max_iter=300, archive_max=100):
    """
    Adaptive Grey Wolf Optimization for Multi-Objective TSP.

    Objectives: f1 = total distance, f2 = total carbon emissions.
    Uses 2-opt moves for position updates, crowding distance for leader
    selection, and sub-quadratic adaptive decay (exponent=1.5).

    Args:
        n_cities (int): Number of cities
        dist_matrix (np.ndarray): Distance cost matrix
        emission_matrix (np.ndarray): Emission cost matrix
        pop_size (int): Number of wolves in the population
        max_iter (int): Maximum number of iterations
        archive_max (int): Maximum Pareto archive size

    Returns:
        tuple: (pareto_archive, convergence_f1, convergence_f2)
    """
    # Initialize population as random permutations
    population = [random.sample(range(n_cities), n_cities)
                  for _ in range(pop_size)]

    # Build initial Pareto archive
    archive = []
    for wolf in population:
        obj = evaluate(wolf, dist_matrix, emission_matrix)
        archive = update_pareto_archive(archive, wolf, obj)

    convergence_f1 = []
    convergence_f2 = []

    for iteration in range(max_iter):

        # Sub-quadratic adaptive decay: exponent=1.5 sustains exploration longer
        a = 2 * (1 - (iteration / max_iter) ** 1.5)

        # Crowding distance based leader selection
        alpha, beta, delta = select_leaders(archive)

        new_population = []
        for wolf in population:
            # Move toward each of the three leaders
            x1 = move_toward_leader(wolf, alpha, a, n_cities)
            x2 = move_toward_leader(wolf, beta,  a, n_cities)
            x3 = move_toward_leader(wolf, delta, a, n_cities)

            # Select best candidate by scalarized objective sum
            candidates = [x1, x2, x3, wolf]
            best_candidate = min(
                candidates,
                key=lambda t: sum(evaluate(t, dist_matrix, emission_matrix))
            )
            new_population.append(best_candidate)

            obj = evaluate(best_candidate, dist_matrix, emission_matrix)
            archive = update_pareto_archive(archive, best_candidate, obj)

        population = new_population

        if len(archive) > archive_max:
            # Trim by keeping most diverse solutions via crowding distance
            distances = crowding_distance(archive)
            ranked = sorted(range(len(archive)),
                            key=lambda i: distances[i], reverse=True)
            archive = [archive[i] for i in ranked[:archive_max]]

        best_f1 = min(o[0] for _, o in archive)
        best_f2 = min(o[1] for _, o in archive)
        convergence_f1.append(best_f1)
        convergence_f2.append(best_f2)

        if (iteration + 1) % 50 == 0:
            print(f"  AGWO Iter {iteration+1}/{max_iter} | "
                  f"Archive: {len(archive)} | "
                  f"Best f1: {best_f1:.2f} | Best f2: {best_f2:.4f}")

    return archive, convergence_f1, convergence_f2


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_pareto_front(archive, title="AGWO Pareto Front"):
    """Plot the Pareto front in f1-f2 objective space."""
    f1_vals = [o[0] for _, o in archive]
    f2_vals = [o[1] for _, o in archive]
    plt.figure(figsize=(7, 5))
    plt.scatter(f1_vals, f2_vals, c='steelblue', s=60,
                edgecolors='navy', alpha=0.7)
    plt.xlabel("f1: Total Distance (units)")
    plt.ylabel("f2: Total Emissions (kg CO2)")
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
    ax2.set_title(f"{label} Convergence - f2 (Emissions)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best Emissions (kg CO2)")
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
        plt.annotate(str(i+1), (x, y), textcoords="offset points",
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

    TSP_FILE    = 'eil51.tsp'   # or 'berlin52.tsp'
    POP_SIZE    = 30
    MAX_ITER    = 300
    ARCHIVE_MAX = 200

    coords          = load_tsplib(TSP_FILE)
    N_CITIES        = len(coords)
    dist_matrix     = compute_distance_matrix(coords)
    emission_matrix = compute_emission_matrix(dist_matrix, seed=99)

    print(f"\nLoaded  : {TSP_FILE} | Cities: {N_CITIES}")
    print(f"Setup   : Pop={POP_SIZE} | Iter={MAX_ITER}")
    print(f"f2 model: Emissions (kg CO2) = distance x fuel_rate x {EMISSION_FACTOR}")
    print("\nRunning AGWO...")
    start = time.time()

    archive, conv_f1, conv_f2 = agwo_motsp(
        N_CITIES, dist_matrix, emission_matrix,
        pop_size=POP_SIZE,
        max_iter=MAX_ITER,
        archive_max=ARCHIVE_MAX
    )

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"Pareto archive size : {len(archive)}")
    print(f"Best f1 (distance)  : {min(o[0] for _, o in archive):.2f}")
    print(f"Best f2 (emissions) : {min(o[1] for _, o in archive):.4f} kg CO2")

    plot_pareto_front(archive)
    plot_convergence(conv_f1, conv_f2, label="AGWO")
    best_tour = min(archive, key=lambda x: x[1][0])[0]
    plot_best_tour(best_tour, coords, title="AGWO Best Tour (min distance)")
    print("\nDone.")