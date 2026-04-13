# TSP-Wizards — Multi-Objective Traveling Salesman Problem

**Course:** DSAI3203 — Fundamentals of Artificial Intelligence  
**Instructor:** Dr. Somaiyeh M.Zadeh  

---

## Overview

This project implements and compares two metaheuristic algorithms — **Adaptive Grey Wolf Optimization (AGWO)** and the **Shuffled Frog Leaping Algorithm (SFLA)** — applied to the Multi-Objective Traveling Salesman Problem (MO-TSP).

The problem is framed with two competing objectives:
- **f1** — Minimize total tour distance (coordinate units)
- **f2** — Minimize total carbon emissions (kg CO2)

Both algorithms maintain a **Pareto archive** of non-dominated solutions throughout their run, producing a Pareto front that represents the best possible trade-offs between distance and emissions.

---

## Algorithms

### AGWO — Adaptive Grey Wolf Optimization
Based on Mirjalili et al. (2014). Key adaptations for MO-TSP:
- **2-opt move operator** replaces continuous position updates
- **Crowding distance leader selection** for α, β, δ wolves
- **Sub-quadratic adaptive decay** (exponent = 1.5) for sustained exploration
- **Pareto archive** with crowding distance trimming

### SFLA — Shuffled Frog Leaping Algorithm
Based on Eusuff & Lansey (2003). Key adaptations for MO-TSP:
- **Position-Based Crossover (PBX)** replaces continuous leap operator
- **Weighted normalized memeplex sorting** (0.6×f1 + 0.4×f2)
- **Pareto archive** updated at each local search step

---

## Project Structure
tsp-wizards/
├── agwo_tsp.py          # AGWO algorithm implementation
├── sfla_tsp.py          # SFLA algorithm implementation
├── benchmark.py         # Benchmark and comparison script
├── .gitignore
├── README.md
├── tsp_container/       # TSPLIB benchmark instances
│   ├── eil51.tsp        # 51 cities — Christofides/Eilon, uniform
│   ├── berlin52.tsp     # 52 cities — Groetschel, clustered
│   ├── eil76.tsp        # 76 cities — Christofides/Eilon, uniform
│   └── st70.tsp         # 70 cities — Smith/Thompson, semi-random
└── results/             # Output images (generated on run)
├── comparison_convergence_<dataset>.png
├── comparison_pareto_<dataset>.png
└── comparison_tours_<dataset>.png

---

## Requirements

Python 3.x with the following libraries:

```bash
pip install numpy matplotlib
```

No other dependencies are required.

---

## How to Run

### Run the full benchmark (recommended)

```bash
python benchmark.py
```

This runs both algorithms 10 times each on the selected dataset and produces:
- A convergence comparison plot
- A Pareto front comparison plot
- A best tour comparison plot
- A summary table printed in the terminal

All images are saved to the `results/` directory.

### Switch datasets

In `benchmark.py`, change the `TSP_FILE` variable at the top:

```python
TSP_FILE = 'eil51.tsp'    # default
TSP_FILE = 'berlin52.tsp'
TSP_FILE = 'eil76.tsp'
TSP_FILE = 'st70.tsp'
```

### Run algorithms individually

```bash
python agwo_tsp.py
python sfla_tsp.py
```

Each script runs on `eil51.tsp` by default and saves individual plots to the root directory.

---

## Configuration

All parameters can be adjusted at the top of `benchmark.py`:

| Parameter | Default | Description |
|---|---|---|
| `N_RUNS` | 10 | Independent runs per algorithm |
| `RANDOM_SEED` | 42 | Base seed (incremented per run) |
| `AGWO_POP_SIZE` | 30 | Number of wolves |
| `AGWO_MAX_ITER` | 300 | Number of iterations |
| `AGWO_ARCHIVE_MAX` | 200 | Max Pareto archive size |
| `SFLA_N_FROGS` | 30 | Number of frogs |
| `SFLA_M_MEMEPLEXES` | 5 | Number of memeplexes |
| `SFLA_N_LOCAL` | 10 | Local iterations per memeplex |
| `SFLA_MAX_SHUFFLES` | 60 | Number of global shuffles |
| `SFLA_ARCHIVE_MAX` | 200 | Max Pareto archive size |

---

## Emission Model

The second objective (f2) uses a fuel-based carbon emission model:
emission(i,j) = distance(i,j) × fuel_rate(i,j) × 2.31

Where:
- `fuel_rate` is randomly assigned per edge between 0.03–0.25 L/km, simulating varying vehicle loads and road conditions
- `2.31` is the standard petrol emission factor in kg CO2 per litre

The emission matrix is generated once per dataset with a fixed seed (99) and shared identically between both algorithms for a fair comparison.

---

## Output

Running `benchmark.py` prints a summary table to the terminal:

========================================================================
BENCHMARK SUMMARY — eil51 (10 runs each)
Metric                    AGWO (mean±std)        SFLA (mean±std)        Winner
Best f1 - Distance        ...                    ...                    ...
Best f2 - Emissions       ...                    ...                    ...
Hypervolume (HV)          ...                    ...                    ...
CPU Time (s)              ...                    ...                    ...
Pareto Archive Size       ...                    ...                    ...

And saves three images to `results/`:

- `comparison_convergence_<dataset>.png` — convergence curves for both algorithms
- `comparison_pareto_<dataset>.png` — overlaid Pareto fronts
- `comparison_tours_<dataset>.png` — best tours side by side on city map

---

## References

[1] S. Mirjalili, S. M. Mirjalili, and A. Lewis, "Grey Wolf Optimizer," *Advances in Engineering Software*, vol. 69, pp. 46–61, 2014.

[2] M. Eusuff and K. Lansey, "Optimization of Water Distribution Network Design Using the Shuffled Frog Leaping Algorithm," *Journal of Water Resources Planning and Management*, vol. 129, no. 3, pp. 210–225, 2003.

[3] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II," *IEEE Trans. Evol. Comput.*, vol. 6, no. 2, pp. 182–197, 2002.

[4] G. Reinelt, "TSPLIB — A Traveling Salesman Problem Library," *ORSA Journal on Computing*, vol. 3, no. 4, pp. 376–384, 1991.

[5] CarbonIndependent.org. (n.d.). *CO2 from fuel use*. https://www.carbonindependent.org/17.html