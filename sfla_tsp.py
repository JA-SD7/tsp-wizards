"""
Shuffled Frog Leaping Algorithm (SFLA) for Multi-Objective TSP
==============================================================
Based on: Eusuff & Lansey (2003) - Shuffled Frog Leaping Algorithm
Adapted for discrete permutation encoding and multi-objective optimization.

Objectives:
    f1 - Minimize total tour distance (km)
    f2 - Minimize total carbon emissions (kg CO2)
         Emission model: distance * fuel_consumption_rate * emission_factor

Usage:
    python sfla_tsp.py

Author: [Your Name]
Course: DSAI3203
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt


# =============================================================================
# CITY / DATASET CONFIGURATION
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
    """Compute Euclidean distance matrix between all city pairs."""
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
# Same emission model as AGWO for a fair comparison.
# f2 = total carbon emissions across the tour.
# Emission per edge = distance * fuel_rate * EMISSION_FACTOR
# EMISSION_FACTOR = 2.31 kg CO2 per litre (standard petrol value)
# =============================================================================
EMISSION_FACTOR = 2.31  # kg CO2 per litre of petrol burned

def compute_emission_matrix(dist_matrix, seed=99):
    """
    Generate a carbon emission matrix based on distance and fuel consumption.

    Each edge is assigned a random fuel consumption rate between 0.06 and
    0.15 L/km simulating varying vehicle loads and road conditions.
    Emission per edge = distance * fuel_rate * EMISSION_FACTOR.

    Args:
        dist_matrix (np.ndarray): Distance matrix
        seed (int): Random seed for reproducibility

    Returns:
        np.ndarray: Emission matrix of shape (n x n), values in kg CO2
    """
    np.random.seed(seed)
    n = len(dist_matrix)
    fuel_rates = np.random.uniform(0.06, 0.15, (n, n))
    emission_matrix = dist_matrix * fuel_rates * EMISSION_FACTOR
    return emission_matrix


def tour_distance(tour, dist_matrix):
    """Calculate total tour distance (f1)."""
    n = len(tour)
    return sum(dist_matrix[tour[i]][tour[(i+1) % n]] for i in range(n))


def tour_emission(tour, emission_matrix):
    """Calculate total carbon emission of a tour (f2)."""
    n = len(tour)
    return sum(emission_matrix[tour[i]][tour[(i+1) % n]] for i in range(n))


def evaluate(tour, dist_matrix, emission_matrix):
    """Evaluate a tour on both objectives. Returns (f1, f2)."""
    return (tour_distance(tour, dist_matrix),
            tour_emission(tour, emission_matrix))


# =============================================================================
# PARETO DOMINANCE & ARCHIVE
# =============================================================================
def dominates(sol_a, sol_b):
    """True if solution a Pareto-dominates solution b."""
    return (sol_a[0] <= sol_b[0] and sol_a[1] <= sol_b[1] and
            (sol_a[0] < sol_b[0] or sol_a[1] < sol_b[1]))


def update_pareto_archive(archive, new_tour, new_obj):
    """Add new solution to Pareto archive if non-dominated."""
    for _, obj in archive:
        if dominates(obj, new_obj):
            return archive
    archive = [(t, o) for t, o in archive if not dominates(new_obj, o)]
    archive.append((new_tour[:], new_obj))
    return archive


# =============================================================================
# [MODIFY ZONE 3] - POSITION-BASED CROSSOVER (PBX) LEAP OPERATOR
# Instead of random swaps, each frog leap uses Position-Based Crossover.
# PBX inherits a random subset of city positions directly from the best
# frog, then fills remaining positions from the current frog in order.
# This is more directed than random swaps — the worst frog actively
# adopts good city placements from the best frog rather than blindly
# shuffling, which leads to faster convergence within each memeplex.
# =============================================================================
def pbx_leap(frog, best_frog, n_cities):
    """
    Perform a Position-Based Crossover leap toward the best frog.

    Inherits a random subset of city positions from best_frog, then
    fills remaining positions with cities from the current frog in
    their original order. Guarantees a valid tour (no duplicates).

    Args:
        frog (list): Current (worst) frog's tour
        best_frog (list): Best frog's tour to leap toward
        n_cities (int): Number of cities

    Returns:
        list: New valid tour after the PBX leap
    """
    # Randomly select how many positions to inherit from best frog
    n_inherit = random.randint(1, n_cities // 2)
    positions = sorted(random.sample(range(n_cities), n_inherit))
    inherited_cities = [best_frog[p] for p in positions]

    # Build new tour: inherited positions fixed, rest filled from frog
    new_tour = [None] * n_cities
    for p, city in zip(positions, inherited_cities):
        new_tour[p] = city

    # Fill remaining slots with cities from frog in original order
    remaining = [c for c in frog if c not in inherited_cities]
    ri = 0
    for i in range(n_cities):
        if new_tour[i] is None:
            new_tour[i] = remaining[ri]
            ri += 1

    return new_tour


# =============================================================================
# [MODIFY ZONE 2 - MEMEPLEX SORT] - WEIGHTED OBJECTIVE SORTING
# Frogs are sorted by a weighted sum of both objectives before being
# distributed into memeplexes. Using 0.6*f1 + 0.4*f2 gives slightly
# more weight to distance (the primary TSP objective) while still
# accounting for emissions. This ensures each memeplex contains frogs
# that are diverse across both objectives, not just ranked by distance.
# =============================================================================
def partition_into_memeplexes(population, objectives, m_memeplexes):
    """
    Sort frogs by weighted objective sum and distribute round-robin
    into m memeplexes.

    Weighted sort key: 0.6*f1 + 0.4*f2 — prioritizes distance slightly
    over emissions while maintaining multi-objective awareness during
    memeplex partitioning.

    Args:
        population (list): List of tours
        objectives (list): List of (f1, f2) tuples
        m_memeplexes (int): Number of memeplexes

    Returns:
        list: List of memeplexes, each a list of (tour, obj) tuples
    """
    # Normalize objectives to avoid scale dominance before weighting
    f1_vals = [o[0] for o in objectives]
    f2_vals = [o[1] for o in objectives]
    f1_max  = max(f1_vals) if max(f1_vals) > 0 else 1
    f2_max  = max(f2_vals) if max(f2_vals) > 0 else 1

    def weighted_key(item):
        _, obj = item
        f1_norm = obj[0] / f1_max
        f2_norm = obj[1] / f2_max
        return 0.6 * f1_norm + 0.4 * f2_norm

    sorted_pop = sorted(zip(population, objectives), key=weighted_key)

    memeplexes = [[] for _ in range(m_memeplexes)]
    for idx, (tour, obj) in enumerate(sorted_pop):
        memeplexes[idx % m_memeplexes].append((tour[:], obj))

    return memeplexes


# =============================================================================
# LOCAL SEARCH WITHIN MEMEPLEX
# =============================================================================
def local_search_memeplex(memeplex, dist_matrix, emission_matrix,
                          archive, n_local, n_cities):
    """
    Perform local search within a single memeplex using PBX leaps.

    In each local iteration:
    1. Worst frog leaps toward local best (Pb) via PBX
    2. If no improvement, leaps toward global best (Pg) via PBX
    3. If still no improvement, replaced with random tour

    Args:
        memeplex (list): List of (tour, obj) tuples
        dist_matrix, emission_matrix: Cost matrices
        archive (list): Global Pareto archive
        n_local (int): Number of local search iterations
        n_cities (int): Number of cities

    Returns:
        tuple: (updated_memeplex, updated_archive)
    """
    for _ in range(n_local):
        if len(memeplex) < 2:
            break

        # Sort by weighted objective to find best and worst
        f1_vals = [o[0] for _, o in memeplex]
        f2_vals = [o[1] for _, o in memeplex]
        f1_max  = max(f1_vals) if max(f1_vals) > 0 else 1
        f2_max  = max(f2_vals) if max(f2_vals) > 0 else 1

        memeplex.sort(
            key=lambda x: 0.6*(x[1][0]/f1_max) + 0.4*(x[1][1]/f2_max)
        )

        pb_tour = memeplex[0][0]   # Best frog in memeplex
        pw_tour = memeplex[-1][0]  # Worst frog in memeplex
        pw_obj  = memeplex[-1][1]

        # Step 1: PBX leap toward local best
        new_tour = pbx_leap(pw_tour, pb_tour, n_cities)
        new_obj  = evaluate(new_tour, dist_matrix, emission_matrix)

        if new_obj[0] < pw_obj[0]:
            memeplex[-1] = (new_tour, new_obj)
            archive = update_pareto_archive(archive, new_tour, new_obj)
            continue

        # Step 2: PBX leap toward global best from archive
        if archive:
            pg_tour  = min(archive, key=lambda x: x[1][0])[0]
            new_tour = pbx_leap(pw_tour, pg_tour, n_cities)
            new_obj  = evaluate(new_tour, dist_matrix, emission_matrix)

            if new_obj[0] < pw_obj[0]:
                memeplex[-1] = (new_tour, new_obj)
                archive = update_pareto_archive(archive, new_tour, new_obj)
                continue

        # Step 3: Replace worst with random tour
        rand_tour = random.sample(range(n_cities), n_cities)
        rand_obj  = evaluate(rand_tour, dist_matrix, emission_matrix)
        memeplex[-1] = (rand_tour, rand_obj)
        archive = update_pareto_archive(archive, rand_tour, rand_obj)

    return memeplex, archive


# =============================================================================
# MAIN SFLA ALGORITHM
# =============================================================================
def sfla_motsp(n_cities, dist_matrix, emission_matrix,
               n_frogs=30, m_memeplexes=5, n_local=10,
               max_shuffles=60, archive_max=100):
    """
    Shuffled Frog Leaping Algorithm for Multi-Objective TSP.

    Objectives: f1 = total distance, f2 = total carbon emissions.
    Uses PBX leap operator and weighted f1+f2 memeplex sorting.

    Args:
        n_cities (int): Number of cities
        dist_matrix (np.ndarray): Distance cost matrix
        emission_matrix (np.ndarray): Emission cost matrix
        n_frogs (int): Total frogs (must be divisible by m_memeplexes)
        m_memeplexes (int): Number of memeplexes
        n_local (int): Local search iterations per memeplex per shuffle
        max_shuffles (int): Number of global shuffles
        archive_max (int): Max Pareto archive size

    Returns:
        tuple: (pareto_archive, convergence_f1, convergence_f2)
    """
    n_frogs  = (n_frogs // m_memeplexes) * m_memeplexes
    q_frogs  = n_frogs // m_memeplexes

    population = [random.sample(range(n_cities), n_cities)
                  for _ in range(n_frogs)]
    objectives = [evaluate(t, dist_matrix, emission_matrix)
                  for t in population]

    archive = []
    for tour, obj in zip(population, objectives):
        archive = update_pareto_archive(archive, tour, obj)

    convergence_f1 = []
    convergence_f2 = []

    print(f"  Config: {n_frogs} frogs | {m_memeplexes} memeplexes | "
          f"{q_frogs} frogs/memeplex")

    for shuffle in range(max_shuffles):

        memeplexes = partition_into_memeplexes(
            population, objectives, m_memeplexes
        )

        updated_memeplexes = []
        for memeplex in memeplexes:
            updated_m, archive = local_search_memeplex(
                memeplex, dist_matrix, emission_matrix,
                archive, n_local, n_cities
            )
            updated_memeplexes.append(updated_m)

        all_frogs  = [frog for m in updated_memeplexes for frog in m]
        population = [t for t, _ in all_frogs]
        objectives = [o for _, o in all_frogs]

        if len(archive) > archive_max:
            archive = archive[:archive_max]

        best_f1 = min(o[0] for _, o in archive)
        best_f2 = min(o[1] for _, o in archive)
        convergence_f1.append(best_f1)
        convergence_f2.append(best_f2)

        if (shuffle + 1) % 10 == 0:
            print(f"  SFLA Shuffle {shuffle+1}/{max_shuffles} | "
                  f"Archive: {len(archive)} | "
                  f"Best f1: {best_f1:.2f} | Best f2: {best_f2:.4f}")

    return archive, convergence_f1, convergence_f2


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_pareto_front(archive, title="SFLA Pareto Front", color='darkorange'):
    """Plot the Pareto front in f1-f2 objective space."""
    f1_vals = [o[0] for _, o in archive]
    f2_vals = [o[1] for _, o in archive]
    plt.figure(figsize=(7, 5))
    plt.scatter(f1_vals, f2_vals, c=color, s=60,
                edgecolors='saddlebrown', alpha=0.7)
    plt.xlabel("f1: Total Distance (units)")
    plt.ylabel("f2: Total Emissions (kg CO2)")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("sfla_pareto_front.png", dpi=150)
    plt.close()
    print("  Saved: sfla_pareto_front.png")


def plot_convergence(conv_f1, conv_f2, label="SFLA"):
    """Plot convergence curves for both objectives."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(conv_f1, color='darkorange')
    ax1.set_title(f"{label} Convergence - f1 (Distance)")
    ax1.set_xlabel("Shuffle")
    ax1.set_ylabel("Best Distance")
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(conv_f2, color='purple')
    ax2.set_title(f"{label} Convergence - f2 (Emissions)")
    ax2.set_xlabel("Shuffle")
    ax2.set_ylabel("Best Emissions (kg CO2)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{label.lower()}_convergence.png", dpi=150)
    plt.close()
    print(f"  Saved: {label.lower()}_convergence.png")


def plot_best_tour(tour, coords, title="Best Tour",
                   filename="sfla_best_tour.png"):
    """Plot the best tour on a 2D city map."""
    tour_coords = [coords[i] for i in tour] + [coords[tour[0]]]
    xs = [c[0] for c in tour_coords]
    ys = [c[1] for c in tour_coords]
    plt.figure(figsize=(7, 7))
    plt.plot(xs, ys, 'o-', color='darkorange', markersize=8)
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=5)
    for i, (x, y) in enumerate(coords):
        plt.annotate(str(i+1), (x, y), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  SFLA for Multi-Objective TSP")
    print("=" * 60)

    TSP_FILE      = 'eil51.tsp'   # or 'berlin52.tsp'
    N_FROGS       = 30
    M_MEMEPLEXES  = 5
    N_LOCAL       = 10
    MAX_SHUFFLES  = 60
    ARCHIVE_MAX   = 100

    coords          = load_tsplib(TSP_FILE)
    N_CITIES        = len(coords)
    dist_matrix     = compute_distance_matrix(coords)
    emission_matrix = compute_emission_matrix(dist_matrix, seed=99)

    print(f"\nLoaded  : {TSP_FILE} | Cities: {N_CITIES}")
    print(f"Setup   : {N_FROGS} frogs | {M_MEMEPLEXES} memeplexes | "
          f"{N_LOCAL} local iters")
    print(f"f2 model: Emissions (kg CO2) = distance x fuel_rate x {EMISSION_FACTOR}")
    print("\nRunning SFLA...")
    start = time.time()

    archive, conv_f1, conv_f2 = sfla_motsp(
        N_CITIES, dist_matrix, emission_matrix,
        n_frogs=N_FROGS,
        m_memeplexes=M_MEMEPLEXES,
        n_local=N_LOCAL,
        max_shuffles=MAX_SHUFFLES,
        archive_max=ARCHIVE_MAX
    )

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"Pareto archive size : {len(archive)}")
    print(f"Best f1 (distance)  : {min(o[0] for _, o in archive):.2f}")
    print(f"Best f2 (emissions) : {min(o[1] for _, o in archive):.4f} kg CO2")

    plot_pareto_front(archive)
    plot_convergence(conv_f1, conv_f2, label="SFLA")
    best_tour = min(archive, key=lambda x: x[1][0])[0]
    plot_best_tour(best_tour, coords,
                   title="SFLA Best Tour (min distance)")
    print("\nDone.")