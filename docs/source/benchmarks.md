# Benchmark Problems

PyGAD ships a small collection of standard benchmark problems under `pygad.benchmarks`. Each problem is a class that can be called with the PyGAD fitness signature `(ga, solution, sol_idx)` and returns a fitness value in PyGAD's maximization format (the original minimization values are negated for you).

Each class also exposes the attributes you usually need to set up the GA:

- `num_genes`: number of decision variables.
- `num_objectives`: number of objectives. `1` for single-objective problems.
- `bounds`: `(low, high)` tuple of variable bounds.

For ZDT problems and ZDT4 / ZDT6, the class also has a `pareto_front(num_points)` method that returns reference points on the true Pareto front. Pass these to the IGD or GD indicators as the `reference_front` argument.

One runnable example per benchmark is available under `examples/benchmarks/`.

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

Available in `pygad.benchmarks.zdt`. All ZDT problems have two objectives and variables in `[0, 1]` (except ZDT4 which uses `[-5, 5]` for the rest of the variables).

| Class | Pareto front shape |
|---|---|
| `ZDT1` | convex |
| `ZDT2` | non-convex |
| `ZDT3` | disconnected (five pieces) |
| `ZDT4` | convex, many local minima in the search space |
| `ZDT6` | non-uniform |

## Many-Objective Problems (DTLZ family)

Available in `pygad.benchmarks.dtlz`. All DTLZ problems support an arbitrary number of objectives `M`. The number of decision variables is `M + k - 1` where `k` is a "distance" variable count.

| Class | Default M | Pareto front shape |
|---|---|---|
| `DTLZ1` | 3 | linear hyperplane (`sum(f_i) = 0.5`) |
| `DTLZ2` | 3 | unit sphere first orthant |
| `DTLZ3` | 3 | unit sphere with hard multimodal g-function |
| `DTLZ4` | 3 | unit sphere with strong bias toward one corner |

## Combinatorial Problems

Two combinatorial benchmarks are available: the 0/1 `Knapsack` and the `TSP`.

### Knapsack

Available in `pygad.benchmarks.knapsack`. The 0/1 `Knapsack` class takes three arguments: a 1D array of item `weights`, a 1D array of item `values`, and a numeric `capacity`. A solution is a binary vector where a 1 means the item is picked. The fitness is the total value when the candidate is within the capacity, and a negative penalty scaled by how much the candidate is over the limit otherwise.

The class exposes `gene_space=[0, 1]` and `gene_type=int` so you can plug it directly into PyGAD:

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

Available in `pygad.benchmarks.tsp`. The `TSP` class can be built from either a 2D array of city `coordinates` or a square `distance_matrix`. A solution is a permutation of the city indices and the fitness is the negative tour length (the tour closes back to the first city). Any non-permutation candidate gets a large negative penalty so the GA keeps a gradient toward feasibility.

The class exposes `gene_space=list(range(num_cities))`, `gene_type=int`, and `allow_duplicate_genes=False` so the permutation constraint is respected:

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

# Measure how close the final population is to the true Pareto front
true_front = problem.pareto_front(num_points=100)
igd = inverted_generational_distance(ga.last_generation_fitness, true_front)
print(f'IGD = {igd}')
```
