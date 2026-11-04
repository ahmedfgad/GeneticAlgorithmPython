"""Public-lifecycle regression tests for no-crossover offspring dimensions."""
import warnings

import numpy
import pytest

import pygad


def fitness_func(ga_instance, solution, solution_idx):
    return float(numpy.sum(solution))


@pytest.mark.parametrize('num_parents,keep_elitism,keep_parents', [
    (2, 5, None),  # More offspring than parents, fewer than elites.
    (6, 0, 3),     # Fewer offspring than parents, more than retained parents.
    (1, 5, None),  # One parent must not silently broadcast into three offspring.
    (2, 8, None),  # All solutions are retained; there are no offspring.
    (6, 0, -1),
    (2, 0, 0),
    (2, 1, None),
])
@pytest.mark.parametrize('mutation_type', [None, 'random'])
def test_no_crossover_preserves_offspring_and_population_dimensions(
        num_parents, keep_elitism, keep_parents, mutation_type):
    initial_population = numpy.arange(24).reshape(8, 3)
    observed_shapes = []

    def on_crossover(ga_instance, offspring):
        observed_shapes.append(offspring.shape)
        assert offspring.shape == (ga_instance.num_offspring, 3)

    params = dict(num_generations=2,
                  num_parents_mating=num_parents,
                  initial_population=initial_population,
                  fitness_func=fitness_func,
                  gene_type=int,
                  parent_selection_type='sss',
                  keep_elitism=keep_elitism,
                  keep_parents=keep_parents,
                  crossover_type=None,
                  mutation_type=mutation_type,
                  on_crossover=on_crossover,
                  random_seed=42)
    if mutation_type is not None:
        params['mutation_probability'] = 0.0
    with warnings.catch_warnings():
        # Retain the library's real validators. The disabled-operators warning
        # is expected for the mutation_type=None cases.
        warnings.simplefilter('ignore', UserWarning)
        ga_instance = pygad.GA(**params)
        ga_instance.run()

    assert ga_instance.generations_completed == 2
    assert ga_instance.population.shape == initial_population.shape
    expected_shape = (ga_instance.num_offspring, 3)
    assert observed_shapes == [expected_shape, expected_shape]
    assert ga_instance.last_generation_offspring_crossover.shape == expected_shape
    assert ga_instance.last_generation_offspring_mutation.shape == expected_shape
    assert all(any(numpy.array_equal(row, original) for original in initial_population)
               for row in ga_instance.population)
    if keep_elitism > 0:
        assert numpy.array_equal(ga_instance.population[0], initial_population[-1])
