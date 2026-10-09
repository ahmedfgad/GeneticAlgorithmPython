# Benchmark Problems

PyGAD bundles common benchmark problems under `pygad.benchmarks`. Each problem is a class callable with `(ga, solution, sol_idx)` and returns a fitness in PyGAD's maximisation format. Minimisation values are negated.

Attributes for setting up the GA (some are class attributes and others are set on each problem instance):

- `num_genes`: number of decision variables.
- `num_objectives`: number of objectives (`1` for single-objective).
- `bounds`: `(low, high)` tuple of variable bounds.

ZDT classes also have a `pareto_front(num_points)` method that returns true-front reference points. Pass these to the IGD or GD indicators as `reference_front`.

Complete scripts are linked beside each benchmark family and listed in the [Examples index](examples.md).

## Single-Objective Problems

Available in `pygad.benchmarks.classic`:

| Class | Global minimum | Bounds |
|---|---|---|
| `Sphere` | f(0, ..., 0) = 0 | `(-5.12, 5.12)` |
| `Rastrigin` | f(0, ..., 0) = 0 | `(-5.12, 5.12)` |
| `Rosenbrock` | f(1, ..., 1) = 0 | `(-5.0, 10.0)` |
| `Griewank` | f(0, ..., 0) = 0 | `(-600.0, 600.0)` |
| `Schwefel` | f(420.97, ..., 420.97) ≈ 0 | `(-500.0, 500.0)` |
| `Ackley` | f(0, ..., 0) = 0 | `(-32.768, 32.768)` |
| `Himmelblau` | four equal minima at f = 0 (2D only) | `(-5.0, 5.0)` |

<!-- python-examples
benchmarks/example_classic_sphere.py
benchmarks/example_classic_rastrigin.py
benchmarks/example_classic_rosenbrock.py
benchmarks/example_classic_griewank.py
benchmarks/example_classic_schwefel.py
benchmarks/example_classic_ackley.py
benchmarks/example_classic_himmelblau.py
-->

**Python examples**

| Python script | What it shows | Related information |
| --- | --- | --- |
| [benchmarks/example_classic_sphere.py](../../examples/benchmarks/example_classic_sphere.py) | **Sphere.** Optimize the Sphere single-objective benchmark. | [Guide](benchmarks.md) |
| [benchmarks/example_classic_rastrigin.py](../../examples/benchmarks/example_classic_rastrigin.py) | **Rastrigin.** Optimize the Rastrigin single-objective benchmark. | [Guide](benchmarks.md) |
| [benchmarks/example_classic_rosenbrock.py](../../examples/benchmarks/example_classic_rosenbrock.py) | **Rosenbrock.** Optimize the Rosenbrock single-objective benchmark. | [Guide](benchmarks.md) |
| [benchmarks/example_classic_griewank.py](../../examples/benchmarks/example_classic_griewank.py) | **Griewank.** Optimize the Griewank single-objective benchmark. | [Guide](benchmarks.md) |
| [benchmarks/example_classic_schwefel.py](../../examples/benchmarks/example_classic_schwefel.py) | **Schwefel.** Optimize the Schwefel single-objective benchmark. | [Guide](benchmarks.md) |
| [benchmarks/example_classic_ackley.py](../../examples/benchmarks/example_classic_ackley.py) | **Ackley.** Optimize the Ackley single-objective benchmark. | [Guide](benchmarks.md) |
| [benchmarks/example_classic_himmelblau.py](../../examples/benchmarks/example_classic_himmelblau.py) | **Himmelblau.** Optimize the Himmelblau single-objective benchmark. | [Guide](benchmarks.md) |

<details>
<summary>Run these examples</summary>

**Sphere** — Requires: PyGAD

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_classic_sphere.py
```

**Rastrigin** — Requires: PyGAD

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_classic_rastrigin.py
```

**Rosenbrock** — Requires: PyGAD

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_classic_rosenbrock.py
```

**Griewank** — Requires: PyGAD

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_classic_griewank.py
```

**Schwefel** — Requires: PyGAD

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_classic_schwefel.py
```

**Ackley** — Requires: PyGAD

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_classic_ackley.py
```

**Himmelblau** — Requires: PyGAD

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_classic_himmelblau.py
```

</details>

<!-- /python-examples -->

## Multi-Objective Problems (ZDT family)

In `pygad.benchmarks.zdt`. Two objectives, variables in `[0, 1]` (ZDT4 uses `[-5, 5]` for the rest).

| Class | Pareto front shape |
|---|---|
| `ZDT1` | convex |
| `ZDT2` | non-convex |
| `ZDT3` | disconnected (five pieces) |
| `ZDT4` | convex, many local minima in the search space |
| `ZDT6` | non-uniform |

<!-- python-examples
benchmarks/example_zdt1.py
benchmarks/example_zdt2.py
benchmarks/example_zdt3.py
benchmarks/example_zdt4.py
benchmarks/example_zdt6.py
-->

**Python examples**

| Python script | What it shows | Related information |
| --- | --- | --- |
| [benchmarks/example_zdt1.py](../../examples/benchmarks/example_zdt1.py) | **ZDT1.** Optimize the ZDT1 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_zdt2.py](../../examples/benchmarks/example_zdt2.py) | **ZDT2.** Optimize the ZDT2 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_zdt3.py](../../examples/benchmarks/example_zdt3.py) | **ZDT3.** Optimize the ZDT3 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_zdt4.py](../../examples/benchmarks/example_zdt4.py) | **ZDT4.** Optimize the ZDT4 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_zdt6.py](../../examples/benchmarks/example_zdt6.py) | **ZDT6.** Optimize the ZDT6 problem and plot its Pareto front. | [Guide](benchmarks.md) |

<details>
<summary>Run these examples</summary>

**ZDT1** — Requires: PyGAD, Matplotlib

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_zdt1.py
```

**ZDT2** — Requires: PyGAD, Matplotlib

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_zdt2.py
```

**ZDT3** — Requires: PyGAD, Matplotlib

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_zdt3.py
```

**ZDT4** — Requires: PyGAD, Matplotlib

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_zdt4.py
```

**ZDT6** — Requires: PyGAD, Matplotlib

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_zdt6.py
```

</details>

<!-- /python-examples -->

## Many-Objective Problems (DTLZ family)

In `pygad.benchmarks.dtlz`. Any number of objectives `M`. Decision variables: `M + k - 1`, where `k` is the distance-variable count.

| Class | Default M | Pareto front shape |
|---|---|---|
| `DTLZ1` | 3 | linear hyperplane (`sum(f_i) = 0.5`) |
| `DTLZ2` | 3 | unit sphere first orthant |
| `DTLZ3` | 3 | unit sphere with hard multimodal g-function |
| `DTLZ4` | 3 | unit sphere with strong bias toward one corner |

<!-- python-examples
benchmarks/example_dtlz1.py
benchmarks/example_dtlz2.py
benchmarks/example_dtlz3.py
benchmarks/example_dtlz4.py
-->

**Python examples**

| Python script | What it shows | Related information |
| --- | --- | --- |
| [benchmarks/example_dtlz1.py](../../examples/benchmarks/example_dtlz1.py) | **DTLZ1.** Optimize the DTLZ1 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_dtlz2.py](../../examples/benchmarks/example_dtlz2.py) | **DTLZ2.** Optimize the DTLZ2 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_dtlz3.py](../../examples/benchmarks/example_dtlz3.py) | **DTLZ3.** Optimize the DTLZ3 problem and plot its Pareto front. | [Guide](benchmarks.md) |
| [benchmarks/example_dtlz4.py](../../examples/benchmarks/example_dtlz4.py) | **DTLZ4.** Optimize the DTLZ4 problem and plot its Pareto front. | [Guide](benchmarks.md) |

<details>
<summary>Run these examples</summary>

**DTLZ1** — Requires: PyGAD

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_dtlz1.py
```

**DTLZ2** — Requires: PyGAD

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_dtlz2.py
```

**DTLZ3** — Requires: PyGAD

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_dtlz3.py
```

**DTLZ4** — Requires: PyGAD

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_dtlz4.py
```

</details>

<!-- /python-examples -->

## Combinatorial Problems

Two combinatorial benchmarks: 0/1 `Knapsack` and `TSP`.

### Knapsack

In `pygad.benchmarks.knapsack`. `Knapsack` takes three arguments: 1D arrays of `weights` and `values`, and a numeric `capacity`. A solution is a binary vector (1 = pick the item). Fitness is the total value within capacity, or a negative penalty scaled by the overweight amount.

Class attributes `gene_space=[0, 1]` and `gene_type=int` plug into PyGAD as is:

```python
import pygad
from pygad.benchmarks.knapsack import Knapsack

problem = Knapsack(weights=[2, 3, 4, 5],
                   values=[3, 4, 5, 6],
                   capacity=5)

ga = pygad.GA(
    num_generations=50,
    num_parents_mating=10,
    fitness_func=problem,
    sol_per_pop=30,
    num_genes=problem.num_genes,
    gene_space=problem.gene_space,
    gene_type=problem.gene_type,
)
ga.run()
```

<!-- python-examples
benchmarks/example_knapsack.py
-->

**Python example**

**[Knapsack](../../examples/benchmarks/example_knapsack.py)**

Select items to maximize value within a weight capacity.

`examples/benchmarks/example_knapsack.py`

<details>
<summary>Run this example</summary>

**Requires:** PyGAD

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_knapsack.py
```

</details>

<!-- /python-examples -->

<!-- sphinx
(tsp-benchmark)=
-->
### Travelling Salesman Problem

In `pygad.benchmarks.tsp`. Build `TSP` from either a 2D `coordinates` array or a square `distance_matrix`. A solution is a permutation of city indices and the fitness is the negative tour length (the tour closes back to the start). Non-permutation candidates get a large negative penalty.

The attributes `gene_space=list(range(num_cities))`, `gene_type=int`, and `allow_duplicate_genes=False` keep the permutation constraint:

#### `TSP` Attributes

- `coordinates`: NumPy array of city coordinates, or `None` when the problem is built from a distance matrix.
- `distance_matrix`: NumPy array of distances between cities, computed from the coordinates or supplied to the constructor.
- `num_genes`: Number of cities, set from the distance matrix's size.
- `gene_space`: List of city indices from `0` through `num_genes - 1`.
- `num_objectives=1`, `gene_type=int`, and `allow_duplicate_genes=False`: Class attributes defining the fitness and permutation encoding.

#### `tour_length(tour)`

Accepts a one-dimensional sequence of city indices and returns the closed tour's length as a Python `float`. Distances are summed between consecutive cities, including the return from the last city to the first. For example, `problem.tour_length([0, 1, 2, 3])` measures that route without calling the fitness function.

This method converts the indices to integers and directly indexes `distance_matrix`; it does not validate that every city is visited exactly once. Pass a valid permutation to measure a TSP solution. The fitness callable `problem(ga, solution, sol_idx)` checks the converted tour's length, uniqueness, and index bounds, returning a negative penalty for invalid tours and `-problem.tour_length(solution)` for valid tours. The `ga` and `sol_idx` arguments are accepted for compatibility with PyGAD and are not used in the calculation.

#### Permutation Mutation

Every city index is already present in a valid tour, so random mutation has no unused replacement value. Its compatible-swap fallback exchanges two city positions instead, keeping the tour valid. Adaptive mutation uses the same fallback. Each position can be swapped at most once in a mutation pass, preventing a second swap from immediately undoing the first.

```python
import pygad
from pygad.benchmarks.tsp import TSP

problem = TSP(coordinates=[[0.0, 0.0],
                           [1.0, 0.0],
                           [1.0, 1.0],
                           [0.0, 1.0]])

ga = pygad.GA(
    num_generations=200,
    num_parents_mating=10,
    fitness_func=problem,
    sol_per_pop=30,
    num_genes=problem.num_genes,
    gene_space=problem.gene_space,
    gene_type=problem.gene_type,
    allow_duplicate_genes=problem.allow_duplicate_genes,
)
ga.run()
```

<!-- python-examples
benchmarks/example_tsp.py
example_travelling_salesman.ipynb
-->

**Python examples**

**[Travelling salesman](../../examples/benchmarks/example_tsp.py)**

Find a short tour using a permutation of four cities.

`examples/benchmarks/example_tsp.py`

<details>
<summary>Run this example</summary>

**Requires:** PyGAD

From the repository root, with the repository version of PyGAD installed:

```console
python examples/benchmarks/example_tsp.py
```

</details>

**[Travelling-salesman Colab notebook](../../examples/example_travelling_salesman.ipynb)**

Explore a city-tour problem using a user-supplied CSV and interactive maps.

`examples/example_travelling_salesman.ipynb`

<details>
<summary>Run this example</summary>

**Requires:** PyGAD, Google Colab, NumPy, pandas, Plotly, folium, and geopy

**Data:** The notebook reads /content/sample_data/startbucks.csv in Google Colab. Supply a compatible CSV at that path. The original data source was not recorded. For local Jupyter use, adapt the Colab-specific imports and CSV path. See the [dataset setup instructions](../../examples/data/README.md).

</details>

<!-- /python-examples -->

## Example: SOO

```python
import pygad
from pygad.benchmarks.classic import Sphere

problem = Sphere(num_genes=10)

ga = pygad.GA(
    num_generations=100,
    num_parents_mating=10,
    fitness_func=problem,
    sol_per_pop=20,
    num_genes=problem.num_genes,
    init_range_low=problem.bounds[0],
    init_range_high=problem.bounds[1],
    crossover_type='sbx',
    sbx_crossover_eta=30,
    mutation_type='polynomial',
    polynomial_mutation_eta=20,
)
ga.run()
```

## Example: MOO

```python
import pygad
from pygad.benchmarks.zdt import ZDT1
from pygad.utils.quality_indicators import inverted_generational_distance

problem = ZDT1(num_genes=10)

ga = pygad.GA(
    num_generations=200,
    num_parents_mating=20,
    fitness_func=problem,
    sol_per_pop=30,
    num_genes=problem.num_genes,
    init_range_low=problem.bounds[0],
    init_range_high=problem.bounds[1],
    parent_selection_type='nsga2',
    crossover_type='sbx',
    sbx_crossover_eta=30,
    mutation_type='polynomial',
    polynomial_mutation_eta=20,
)
ga.run()

# IGD against the true front.
true_front = problem.pareto_front(num_points=100)
igd = inverted_generational_distance(ga.last_generation_fitness, true_front)
print(f'IGD = {igd}')
```
