"""
Tests for the NSGA-III auto-growth path in the engine.

When ``sol_per_pop`` is smaller than the number of NSGA-III reference
points, the engine grows the population to match before the
generational loop starts. The grown rows must follow every rule that
applies to the initial population: per-gene init range,
``gene_space``, ``gene_type`` (single or nested),
``allow_duplicate_genes``, and ``gene_constraint``.

The tests here exercise each rule with a population of size 1 so the
auto-growth path is forced to generate fresh solutions.
"""

import numpy
import pytest

import pygad


def _two_objective_fitness(ga, solution, sol_idx):
    return [float(numpy.sum(solution)), -float(numpy.sum(numpy.asarray(solution) ** 2))]


def _three_objective_fitness(ga, solution, sol_idx):
    return [float(solution[0]), float(solution[1]), float(solution[2])]


def _build_ga_and_grow(**kwargs):
    """
    Create a GA with sol_per_pop small enough to trigger NSGA-III auto-
    growth and run a single generation so the population is grown and
    re-evaluated before the test assertion runs.
    """
    defaults = dict(
        num_generations=1,
        num_parents_mating=3,
        fitness_func=_three_objective_fitness,
        sol_per_pop=4,
        num_genes=4,
        parent_selection_type='nsga3',
        nsga3_num_divisions=4,
        random_seed=7,
        suppress_warnings=True,
    )
    defaults.update(kwargs)
    ga = pygad.GA(**defaults)
    ga.run()
    return ga


def test_population_growth_respects_init_range_low_and_high():
    # With per-gene init range, every gene in every grown row must sit
    # inside its own [low, high] window.
    init_range_low = [0.0, 1.0, 2.0, 3.0]
    init_range_high = [0.5, 1.5, 2.5, 3.5]
    ga = _build_ga_and_grow(
        num_genes=4,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
    )
    population = numpy.asarray(ga.initial_population, dtype=float)
    for gene_idx in range(population.shape[1]):
        column = population[:, gene_idx]
        assert column.min() >= init_range_low[gene_idx]
        assert column.max() <= init_range_high[gene_idx]


def test_population_growth_respects_gene_space_discrete_values():
    # Every gene in every row must come from the discrete gene space.
    allowed = [10, 20, 30]
    ga = _build_ga_and_grow(
        num_genes=4,
        gene_type=int,
        gene_space=allowed,
    )
    population = numpy.asarray(ga.initial_population, dtype=int)
    assert set(population.flatten().tolist()).issubset(set(allowed))


def test_population_growth_respects_single_gene_type_int():
    # gene_type=int means every gene in the grown rows must be integer.
    ga = _build_ga_and_grow(
        num_genes=4,
        gene_type=int,
    )
    population = numpy.asarray(ga.initial_population)
    assert population.dtype == int


def test_population_growth_respects_nested_gene_types_per_gene():
    # Mixed dtypes per gene (and a precision for the float gene). The
    # int gene must round-trip to an int, the float gene must have at
    # most 2 decimals.
    gene_type = [int, [float, 2], int, [float, 3]]
    ga = _build_ga_and_grow(
        num_genes=4,
        gene_type=gene_type,
    )
    population = ga.initial_population
    for sol_idx in range(population.shape[0]):
        assert isinstance(population[sol_idx, 0], (int, numpy.integer))
        assert isinstance(population[sol_idx, 2], (int, numpy.integer))
        float_gene_one = float(population[sol_idx, 1])
        float_gene_two = float(population[sol_idx, 3])
        assert round(float_gene_one, 2) == float_gene_one
        assert round(float_gene_two, 3) == float_gene_two


def test_population_growth_respects_allow_duplicate_genes_false():
    # When duplicates are not allowed, no row may contain the same gene
    # value twice.
    ga = _build_ga_and_grow(
        num_genes=4,
        gene_type=int,
        gene_space=list(range(20)),
        allow_duplicate_genes=False,
    )
    population = numpy.asarray(ga.initial_population, dtype=int)
    for row in population:
        assert len(set(row.tolist())) == len(row)


def test_population_growth_respects_gene_constraint_callable():
    # Constraint forces gene 0 to be >= 5 and gene 1 to be even. Every
    # grown row must satisfy both constraints.
    def gene_zero_at_least_five(solution, values):
        return [v for v in values if v >= 5]

    def gene_one_must_be_even(solution, values):
        return [v for v in values if int(v) % 2 == 0]

    ga = _build_ga_and_grow(
        num_genes=4,
        gene_type=int,
        gene_space=list(range(20)),
        gene_constraint=[gene_zero_at_least_five,
                         gene_one_must_be_even,
                         None,
                         None],
    )
    population = numpy.asarray(ga.initial_population, dtype=int)
    assert (population[:, 0] >= 5).all()
    assert (population[:, 1] % 2 == 0).all()


def test_generate_single_random_gene_uses_initial_population_range_not_mutation_range():
    # With gene_space=None the helper must sample from
    # [init_range_low, init_range_high], not from
    # [random_mutation_min_val, random_mutation_max_val]. Set the two
    # windows to non-overlapping ranges so any leak from the mutation
    # range is detectable. Seed numpy so the helper's draw is
    # deterministic.
    ga = pygad.GA(
        num_generations=1,
        num_parents_mating=3,
        fitness_func=_two_objective_fitness,
        sol_per_pop=4,
        num_genes=2,
        init_range_low=100.0,
        init_range_high=200.0,
        random_mutation_min_val=-200.0,
        random_mutation_max_val=-100.0,
        parent_selection_type='nsga2',
        random_seed=11,
        suppress_warnings=True,
    )
    numpy.random.seed(11)
    drawn_values = [ga._nsga3_generate_single_random_gene(gene_idx=0, partial_solution=numpy.empty(2, dtype=object))
                    for _ in range(50)]
    drawn = numpy.asarray(drawn_values, dtype=float).flatten()
    assert drawn.min() >= 100.0
    assert drawn.max() <= 200.0


def test_generate_single_random_gene_uses_gene_space_when_present():
    # With gene_space set the helper must draw from the gene space, not
    # from the init range. Use disjoint init range and gene space so a
    # leak is detectable.
    allowed = [50, 51, 52]
    ga = pygad.GA(
        num_generations=1,
        num_parents_mating=3,
        fitness_func=_two_objective_fitness,
        sol_per_pop=4,
        num_genes=2,
        gene_type=int,
        gene_space=allowed,
        init_range_low=-100,
        init_range_high=-50,
        parent_selection_type='nsga2',
        random_seed=11,
        suppress_warnings=True,
    )
    numpy.random.seed(13)
    drawn_values = [int(ga._nsga3_generate_single_random_gene(gene_idx=0, partial_solution=numpy.empty(2, dtype=object)))
                    for _ in range(50)]
    assert set(drawn_values).issubset(set(allowed))
