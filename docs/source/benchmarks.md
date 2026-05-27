# Benchmark Problems

PyGAD bundles common benchmark problems under `pygad.benchmarks`. Each problem is a class callable with `(ga, solution, sol_idx)` and returns a fitness in PyGAD's maximisation format. Minimisation values are negated.

Class attributes for setting up the GA:

- `num_genes`: number of decision variables.
- `num_objectives`: number of objectives (`1` for single-objective).
- `bounds`: `(low, high)` tuple of variable bounds.

ZDT classes also have a `pareto_front(num_points)` method that returns true-front reference points. Pass these to the IGD or GD indicators as `reference_front`.

A runnable example per benchmark lives under `examples/benchmarks/`.

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

## Multi-Objective Problems (ZDT family)

In `pygad.benchmarks.zdt`. Two objectives, variables in `[0, 1]` (ZDT4 uses `[-5, 5]` for the rest).

| Class | Pareto front shape |
|---|---|
| `ZDT1` | convex |
| `ZDT2` | non-convex |
| `ZDT3` | disconnected (five pieces) |
| `ZDT4` | convex, many local minima in the search space |
| `ZDT6` | non-uniform |

## Many-Objective Problems (DTLZ family)

In `pygad.benchmarks.dtlz`. Any number of objectives `M`. Decision variables: `M + k - 1`, where `k` is the distance-variable count.

| Class | Default M | Pareto front shape |
|---|---|---|
| `DTLZ1` | 3 | linear hyperplane (`sum(f_i) = 0.5`) |
| `DTLZ2` | 3 | unit sphere first orthant |
| `DTLZ3` | 3 | unit sphere with hard multimodal g-function |
| `DTLZ4` | 3 | unit sphere with strong bias toward one corner |

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

### Travelling Salesman Problem

In `pygad.benchmarks.tsp`. Build `TSP` from either a 2D `coordinates` array or a square `distance_matrix`. A solution is a permutation of city indices and the fitness is the negative tour length (the tour closes back to the start). Non-permutation candidates get a large negative penalty.

Class attributes `gene_space=list(range(num_cities))`, `gene_type=int`, and `allow_duplicate_genes=False` keep the permutation constraint:

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
