import numpy
import pytest

import pygad


def _sum_fitness(ga, solution, sol_idx):
    return float(numpy.sum(solution))


def _make_ga(crossover_type="sbx", mutation_type="polynomial",
             sbx_crossover_eta=30, polynomial_mutation_eta=20,
             init_range_low=0.0, init_range_high=1.0,
             num_generations=10, sol_per_pop=12, num_genes=4):
    return pygad.GA(num_generations=num_generations,
                    num_parents_mating=6,
                    fitness_func=_sum_fitness,
                    sol_per_pop=sol_per_pop,
                    num_genes=num_genes,
                    init_range_low=init_range_low,
                    init_range_high=init_range_high,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    sbx_crossover_eta=sbx_crossover_eta,
                    polynomial_mutation_eta=polynomial_mutation_eta,
                    random_seed=42,
                    suppress_warnings=True)


def test_sbx_crossover_runs_and_keeps_population_shape():
    ga = _make_ga(mutation_type=None)
    ga.run()
    assert ga.population.shape == (12, 4)


def test_polynomial_mutation_runs_and_keeps_population_shape():
    ga = _make_ga(crossover_type="single_point")
    ga.run()
    assert ga.population.shape == (12, 4)


def test_sbx_offspring_stay_within_bounds():
    # SBX is bounded; offspring must never leave [init_range_low,
    # init_range_high].
    ga = _make_ga(mutation_type=None,
                  init_range_low=-2.0,
                  init_range_high=2.0)
    ga.run()
    pop = numpy.asarray(ga.population, dtype=float)
    assert pop.min() >= -2.0 - 1e-9
    assert pop.max() <= 2.0 + 1e-9


def test_polynomial_mutation_offspring_stay_within_bounds():
    # Polynomial mutation is bounded; the mutated value must never leave
    # the per-gene bounds.
    ga = _make_ga(crossover_type="single_point",
                  init_range_low=-5.0,
                  init_range_high=5.0)
    ga.run()
    pop = numpy.asarray(ga.population, dtype=float)
    assert pop.min() >= -5.0 - 1e-9
    assert pop.max() <= 5.0 + 1e-9


def test_sbx_high_eta_keeps_children_close_to_parents():
    # A very large eta makes the SBX spread factor collapse so the
    # child is essentially one of the parents on every gene (within
    # numerical noise). Verify the offspring values are within the
    # closed interval defined by the two parents on each gene.
    nsga = pygad.GA(num_generations=1, num_parents_mating=2,
                    fitness_func=_sum_fitness,
                    sol_per_pop=4, num_genes=2,
                    init_range_low=0.0, init_range_high=1.0,
                    crossover_type='sbx', mutation_type=None,
                    sbx_crossover_eta=1e6,
                    random_seed=1, suppress_warnings=True)
    parents = numpy.array([[0.2, 0.4], [0.6, 0.8]])
    offspring = nsga.sbx_crossover(parents, (1, 2))
    lower = numpy.minimum(parents[0], parents[1])
    upper = numpy.maximum(parents[0], parents[1])
    assert numpy.all(offspring[0] >= lower - 1e-6)
    assert numpy.all(offspring[0] <= upper + 1e-6)


def test_polynomial_mutation_high_eta_makes_small_steps():
    # A very large eta should produce a mutated value almost equal to
    # the input value because the polynomial step collapses to ~0.
    ga = pygad.GA(num_generations=1, num_parents_mating=2,
                  fitness_func=_sum_fitness,
                  sol_per_pop=4, num_genes=3,
                  init_range_low=0.0, init_range_high=1.0,
                  crossover_type=None, mutation_type='polynomial',
                  polynomial_mutation_eta=1e6,
                  mutation_probability=1.0,
                  random_seed=7, suppress_warnings=True)
    offspring = numpy.array([[0.3, 0.5, 0.7]])
    mutated = ga.polynomial_mutation(offspring.copy())
    numpy.testing.assert_allclose(mutated, offspring, atol=1e-3)


def test_sbx_crossover_eta_validation():
    with pytest.raises(ValueError, match="sbx_crossover_eta"):
        pygad.GA(num_generations=2, num_parents_mating=2,
                 fitness_func=_sum_fitness, sol_per_pop=4, num_genes=2,
                 crossover_type='sbx', sbx_crossover_eta=-1,
                 suppress_warnings=True)


def test_polynomial_mutation_eta_validation():
    with pytest.raises(ValueError, match="polynomial_mutation_eta"):
        pygad.GA(num_generations=2, num_parents_mating=2,
                 fitness_func=_sum_fitness, sol_per_pop=4, num_genes=2,
                 mutation_type='polynomial', polynomial_mutation_eta=0,
                 suppress_warnings=True)


def test_unknown_crossover_error_message_lists_sbx():
    # Make sure the user-facing error message mentions sbx as a valid
    # option.
    with pytest.raises(TypeError, match="sbx"):
        pygad.GA(num_generations=2, num_parents_mating=2,
                 fitness_func=_sum_fitness, sol_per_pop=4, num_genes=2,
                 crossover_type='bogus', suppress_warnings=True)


def test_unknown_mutation_error_message_lists_polynomial():
    with pytest.raises(TypeError, match="polynomial"):
        pygad.GA(num_generations=2, num_parents_mating=2,
                 fitness_func=_sum_fitness, sol_per_pop=4, num_genes=2,
                 mutation_type='bogus', suppress_warnings=True)


def test_sbx_with_fixed_seed_matches_pinned_output():
    # Pinned regression: with numpy.random seeded to 0 and the two
    # parents below, the SBX formula must produce exactly these
    # offspring values. The expected values come from the standard
    # Deb-Beyer bounded SBX formula on these inputs.
    ga = pygad.GA(num_generations=1, num_parents_mating=2,
                  fitness_func=_sum_fitness, sol_per_pop=4, num_genes=4,
                  init_range_low=0.0, init_range_high=1.0,
                  crossover_type='sbx', sbx_crossover_eta=30,
                  suppress_warnings=True)
    parents = numpy.array([
        [0.2, 0.5, 0.7, 0.9],
        [0.4, 0.3, 0.5, 0.1],
    ])
    numpy.random.seed(0)
    offspring = ga.sbx_crossover(parents, (2, 4))
    expected = numpy.array([
        [0.19966807185839858, 0.29816798996651034, 0.49925505848096274, 0.0987922279628985],
        [0.20053305524715195, 0.29888084145301086, 0.500429179839271, 0.07981285137676747],
    ])
    numpy.testing.assert_allclose(offspring, expected, atol=1e-12)


def test_polynomial_mutation_with_fixed_seed_matches_pinned_output():
    # Pinned regression: with numpy.random seeded to 0 and the input
    # vector below, polynomial mutation must produce exactly these
    # values. The expected values come from the standard Deb 1996
    # bounded polynomial mutation formula on these inputs.
    ga = pygad.GA(num_generations=1, num_parents_mating=2,
                  fitness_func=_sum_fitness, sol_per_pop=4, num_genes=4,
                  init_range_low=0.0, init_range_high=1.0,
                  mutation_type='polynomial', polynomial_mutation_eta=20,
                  mutation_probability=1.0, suppress_warnings=True)
    numpy.random.seed(0)
    mutated = ga.polynomial_mutation(numpy.array([[0.5, 0.5, 0.5, 0.5]], dtype=float))
    expected = numpy.array([[0.5264432889397432, 0.5044687436392203,
                             0.5162949167026651, 0.5702829850254326]])
    numpy.testing.assert_allclose(mutated, expected, atol=1e-12)
