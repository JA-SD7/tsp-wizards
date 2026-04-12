"""
Benchmark & Comparison Script for AGWO vs SFLA on Multi-Objective TSP
======================================================================
Runs both algorithms on the same dataset, collects results, and produces
a full comparative analysis including:
  - Convergence curves (both algorithms on same plot, normalized x-axis)
  - Pareto front comparison (overlaid scatter plot)
  - Summary comparison table (distance, time, hypervolume, CPU time)

Usage:
    python benchmark.py

Author: [Your Name]
Course: DSAI3203
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt

# Import both algorithm main functions.
# Utility functions are imported from agwo_tsp to avoid duplication.
# Make sure your files are named exactly: agwo_tsp.py and sfla_tsp.py
from agwo_tsp import (
    agwo_motsp,
    load_tsplib,
    compute_distance_matrix,
    compute_time_matrix,
)
from sfla_tsp import sfla_motsp


# =============================================================================
# CONFIGURATION
# =============================================================================
TSP_FILE          = 'eil51.tsp'   # or 'berlin52.tsp'
N_RUNS            = 10            # Independent runs per algorithm
RANDOM_SEED       = 42            # Base seed (incremented per run)

# AGWO parameters
AGWO_POP_SIZE     = 30
AGWO_MAX_ITER     = 300
AGWO_ARCHIVE_MAX  = 100

# SFLA parameters
SFLA_N_FROGS      = 30
SFLA_M_MEMEPLEXES = 5
SFLA_N_LOCAL      = 10
SFLA_MAX_SHUFFLES = 60
SFLA_ARCHIVE_MAX  = 100

# Number of points to normalize convergence curves to (for fair comparison)
NORM_POINTS       = 100


# =============================================================================
# HYPERVOLUME INDICATOR
# =============================================================================
def compute_hypervolume(archive, ref_point):
    """
    Compute the 2D hypervolume indicator for a Pareto front.
    Uses a sweep-line method. Higher = better Pareto front coverage.

    Args:
        archive (list): List of (tour, (f1, f2)) tuples
        ref_point (tuple): (ref_f1, ref_f2) worse than all solutions

    Returns:
        float: Hypervolume value
    """
    points = sorted([o for _, o in archive], key=lambda x: x[0])
    hv = 0.0
    prev_f1 = ref_point[0]
    for f1, f2 in reversed(points):
        width = prev_f1 - f1
        height = ref_point[1] - f2
        if width > 0 and height > 0:
            hv += width * height
        prev_f1 = f1
    return hv


def get_reference_point(all_archives, margin=1.1):
    """
    Compute a reference point slightly worse than the worst solution
    across all archives combined.

    Args:
        all_archives (list): All archives from all runs and algorithms
        margin (float): Multiplicative margin above worst values

    Returns:
        tuple: (ref_f1, ref_f2)
    """
    all_f1 = [o[0] for archive in all_archives for _, o in archive]
    all_f2 = [o[1] for archive in all_archives for _, o in archive]
    return (max(all_f1) * margin, max(all_f2) * margin)


# =============================================================================
# NORMALIZE CONVERGENCE
# Resamples a convergence curve to a fixed number of points (0-100%)
# so curves of different lengths can be fairly compared on the same axis.
# =============================================================================
def normalize_convergence(curve, n_points=NORM_POINTS):
    """
    Resample a convergence curve to n_points evenly spaced values.

    Args:
        curve (list): Original convergence values (any length)
        n_points (int): Number of points to resample to

    Returns:
        np.ndarray: Resampled curve of length n_points
    """
    curve = np.array(curve)
    original_x = np.linspace(0, 100, len(curve))
    new_x = np.linspace(0, 100, n_points)
    return np.interp(new_x, original_x, curve)


# =============================================================================
# MULTI-RUN EXECUTOR
# =============================================================================
def run_algorithm(name, algo_func, algo_kwargs, n_cities,
                  dist_matrix, time_matrix, n_runs, base_seed):
    """
    Run an algorithm multiple times and collect performance statistics.

    Args:
        name (str): Algorithm name for display
        algo_func (callable): The algorithm function (agwo_motsp or sfla_motsp)
        algo_kwargs (dict): Keyword arguments passed to the algorithm
        n_cities (int): Number of cities
        dist_matrix, time_matrix: Cost matrices
        n_runs (int): Number of independent runs
        base_seed (int): Base random seed (incremented per run)

    Returns:
        dict: Results containing archives, normalized convergence, and timing
    """
    all_archives      = []
    all_conv_f1_norm  = []
    all_conv_f2_norm  = []
    all_best_f1       = []
    all_best_f2       = []
    all_times         = []

    print(f"\n{'='*60}")
    print(f"  Running {name} ({n_runs} runs)")
    print(f"{'='*60}")

    for run in range(n_runs):
        random.seed(base_seed + run)
        np.random.seed(base_seed + run)

        start = time.time()
        archive, conv_f1, conv_f2 = algo_func(
            n_cities, dist_matrix, time_matrix, **algo_kwargs
        )
        elapsed = time.time() - start

        best_f1 = min(o[0] for _, o in archive)
        best_f2 = min(o[1] for _, o in archive)

        all_archives.append(archive)
        all_conv_f1_norm.append(normalize_convergence(conv_f1))
        all_conv_f2_norm.append(normalize_convergence(conv_f2))
        all_best_f1.append(best_f1)
        all_best_f2.append(best_f2)
        all_times.append(elapsed)

        print(f"  Run {run+1:02d}/{n_runs} | "
              f"f1: {best_f1:.2f} | f2: {best_f2:.4f} | "
              f"Archive: {len(archive)} | Time: {elapsed:.2f}s")

    return {
        'name':     name,
        'archives': all_archives,
        'conv_f1':  all_conv_f1_norm,
        'conv_f2':  all_conv_f2_norm,
        'best_f1':  all_best_f1,
        'best_f2':  all_best_f2,
        'times':    all_times,
    }


# =============================================================================
# PLOTS
# =============================================================================
def plot_convergence_comparison(agwo_results, sfla_results, dataset_name):
    """
    Plot AGWO vs SFLA convergence curves on the same axes.
    X-axis is normalized to 0-100% progress so both algorithms are
    fairly comparable regardless of their different loop counts.
    Shows mean curve with shaded min/max band across all runs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Convergence Comparison — {dataset_name}", fontsize=13)

    x = np.linspace(0, 100, NORM_POINTS)

    for ax, obj_key, ylabel, title_suffix in [
        (ax1, 'conv_f1', 'Best Distance', 'f1 (Distance)'),
        (ax2, 'conv_f2', 'Best Time',     'f2 (Travel Time)')
    ]:
        for results, color, label in [
            (agwo_results, 'steelblue',  'AGWO'),
            (sfla_results, 'darkorange', 'SFLA')
        ]:
            runs = np.array(results[obj_key])
            mean = runs.mean(axis=0)
            mn   = runs.min(axis=0)
            mx   = runs.max(axis=0)

            ax.plot(x, mean, color=color, label=label, linewidth=2)
            ax.fill_between(x, mn, mx, color=color, alpha=0.15)

        ax.set_title(f"Convergence — {title_suffix}")
        ax.set_xlabel("Progress (%)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fname = f"comparison_convergence_{dataset_name}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"\n  Saved: {fname}")


def plot_pareto_comparison(agwo_results, sfla_results, dataset_name):
    """
    Overlay AGWO and SFLA Pareto fronts from their best run on one plot.
    """
    agwo_best_idx = int(np.argmin(agwo_results['best_f1']))
    sfla_best_idx = int(np.argmin(sfla_results['best_f1']))

    agwo_archive = agwo_results['archives'][agwo_best_idx]
    sfla_archive = sfla_results['archives'][sfla_best_idx]

    agwo_f1 = [o[0] for _, o in agwo_archive]
    agwo_f2 = [o[1] for _, o in agwo_archive]
    sfla_f1 = [o[0] for _, o in sfla_archive]
    sfla_f2 = [o[1] for _, o in sfla_archive]

    plt.figure(figsize=(8, 6))
    plt.scatter(agwo_f1, agwo_f2, c='steelblue', s=60,
                edgecolors='navy', alpha=0.7, label='AGWO')
    plt.scatter(sfla_f1, sfla_f2, c='darkorange', s=60,
                edgecolors='saddlebrown', alpha=0.7, label='SFLA')
    plt.xlabel("f1: Total Distance")
    plt.ylabel("f2: Total Travel Time")
    plt.title(f"Pareto Front Comparison — {dataset_name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fname = f"comparison_pareto_{dataset_name}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_best_tours_side_by_side(agwo_results, sfla_results, coords, dataset_name):
    """
    Plot the best tour found by each algorithm side by side on a city map.
    """
    agwo_best_idx = int(np.argmin(agwo_results['best_f1']))
    sfla_best_idx = int(np.argmin(sfla_results['best_f1']))

    agwo_tour = min(agwo_results['archives'][agwo_best_idx],
                    key=lambda x: x[1][0])[0]
    sfla_tour = min(sfla_results['archives'][sfla_best_idx],
                    key=lambda x: x[1][0])[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Best Tours — {dataset_name}", fontsize=13)

    for ax, tour, label, color in [
        (ax1, agwo_tour, 'AGWO', 'steelblue'),
        (ax2, sfla_tour, 'SFLA', 'darkorange')
    ]:
        route = [coords[i] for i in tour] + [coords[tour[0]]]
        xs = [c[0] for c in route]
        ys = [c[1] for c in route]
        ax.plot(xs, ys, 'o-', color=color, markersize=6, linewidth=1.5)
        ax.scatter(coords[:, 0], coords[:, 1], c='red', s=80, zorder=5)
        for i, (x, y) in enumerate(coords):
            ax.annotate(str(i+1), (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
        ax.set_title(f"{label} Best Tour")
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    fname = f"comparison_tours_{dataset_name}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# =============================================================================
# SUMMARY TABLE
# =============================================================================
def print_summary_table(agwo_results, sfla_results, ref_point, dataset_name):
    """
    Print a formatted comparison table of all key metrics.
    Reports mean ± std across all runs for each algorithm.
    """
    def stats(values):
        return np.mean(values), np.std(values)

    agwo_hvs   = [compute_hypervolume(a, ref_point)
                  for a in agwo_results['archives']]
    sfla_hvs   = [compute_hypervolume(a, ref_point)
                  for a in sfla_results['archives']]
    agwo_sizes = [len(a) for a in agwo_results['archives']]
    sfla_sizes = [len(a) for a in sfla_results['archives']]

    # (label, agwo_values, sfla_values, lower_is_better)
    metrics = [
        ("Best f1 - Distance",  agwo_results['best_f1'], sfla_results['best_f1'], True),
        ("Best f2 - Time",      agwo_results['best_f2'], sfla_results['best_f2'], True),
        ("Hypervolume (HV)",    agwo_hvs,                sfla_hvs,                False),
        ("CPU Time (s)",        agwo_results['times'],   sfla_results['times'],   True),
        ("Pareto Archive Size", agwo_sizes,              sfla_sizes,              False),
    ]

    print(f"\n{'='*72}")
    print(f"  BENCHMARK SUMMARY — {dataset_name} ({N_RUNS} runs each)")
    print(f"{'='*72}")
    print(f"  {'Metric':<25} {'AGWO (mean±std)':<22} {'SFLA (mean±std)':<22} {'Winner'}")
    print(f"  {'-'*68}")

    for label, agwo_vals, sfla_vals, lower_is_better in metrics:
        am, astd = stats(agwo_vals)
        sm, sstd = stats(sfla_vals)
        winner = 'AGWO' if (am < sm) == lower_is_better else 'SFLA'
        print(f"  {label:<25} {am:>10.4f} ±{astd:>8.4f}   "
              f"{sm:>10.4f} ±{sstd:>8.4f}   {winner}")

    print(f"{'='*72}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  AGWO vs SFLA — Benchmark & Comparison")
    print("=" * 60)

    # --- Load dataset once, shared by both algorithms ---
    coords       = load_tsplib(TSP_FILE)
    N_CITIES     = len(coords)
    dist_matrix  = compute_distance_matrix(coords)
    time_matrix  = compute_time_matrix(dist_matrix, seed=99)
    dataset_name = TSP_FILE.replace('.tsp', '')

    print(f"\nDataset : {TSP_FILE}")
    print(f"Cities  : {N_CITIES}")
    print(f"Runs    : {N_RUNS} per algorithm")

    # --- Run AGWO ---
    agwo_results = run_algorithm(
        name        = 'AGWO',
        algo_func   = agwo_motsp,
        algo_kwargs = {
            'pop_size':    AGWO_POP_SIZE,
            'max_iter':    AGWO_MAX_ITER,
            'archive_max': AGWO_ARCHIVE_MAX,
        },
        n_cities    = N_CITIES,
        dist_matrix = dist_matrix,
        time_matrix = time_matrix,
        n_runs      = N_RUNS,
        base_seed   = RANDOM_SEED
    )

    # --- Run SFLA ---
    sfla_results = run_algorithm(
        name        = 'SFLA',
        algo_func   = sfla_motsp,
        algo_kwargs = {
            'n_frogs':      SFLA_N_FROGS,
            'm_memeplexes': SFLA_M_MEMEPLEXES,
            'n_local':      SFLA_N_LOCAL,
            'max_shuffles': SFLA_MAX_SHUFFLES,
            'archive_max':  SFLA_ARCHIVE_MAX,
        },
        n_cities    = N_CITIES,
        dist_matrix = dist_matrix,
        time_matrix = time_matrix,
        n_runs      = N_RUNS,
        base_seed   = RANDOM_SEED
    )

    # --- Shared reference point for hypervolume (must see all archives) ---
    all_archives = agwo_results['archives'] + sfla_results['archives']
    ref_point    = get_reference_point(all_archives)

    # --- Print summary table ---
    print_summary_table(agwo_results, sfla_results, ref_point, dataset_name)

    # --- Generate and save all plots ---
    print("  Generating plots...")
    plot_convergence_comparison(agwo_results, sfla_results, dataset_name)
    plot_pareto_comparison(agwo_results, sfla_results, dataset_name)
    plot_best_tours_side_by_side(agwo_results, sfla_results, coords, dataset_name)

    print("\nAll done. Output files saved to current directory.")