"""Random mutation respects NumPy integer limits before converting candidates."""

import numpy
import pytest

import pygad


def fitness_func(ga_instance, solution, solution_index):
    return float(sum(int(value) for value in solution))


def make_ga(**options):
    parameters = dict(num_generations=2, num_parents_mating=2,
                      fitness_func=fitness_func, sol_per_pop=4, num_genes=3,
                      gene_type=numpy.uint8, init_range_low=0, init_range_high=3,
                      random_seed=7, suppress_warnings=True)
    parameters.update(options)
    return pygad.GA(**parameters)


integer_types = [numpy.int8, numpy.int16, numpy.int32, numpy.int64,
                 numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]


@pytest.mark.parametrize("gene_type", integer_types)
@pytest.mark.parametrize("mutation_by_replacement", [False, True])
def test_mutation_clamps_scalar_and_array_values_at_both_integer_limits(gene_type, mutation_by_replacement):
    ga_instance = make_ga(gene_type=gene_type)
    type_limits = numpy.iinfo(gene_type)
    for boundary, offset in [(int(type_limits.min), -1), (int(type_limits.max), 1)]:
        random_value = boundary + offset if mutation_by_replacement else offset
        converted = ga_instance.mutation_change_gene_dtype_and_round(
            random_value, 0, gene_type(boundary), mutation_by_replacement)
        assert type(converted) is gene_type
        assert int(converted) == boundary

        random_values = numpy.array([[random_value, boundary if mutation_by_replacement else 0]], dtype=object)
        original_values = random_values.copy()
        converted = ga_instance.mutation_change_gene_dtype_and_round(
            random_values, 0, gene_type(boundary), mutation_by_replacement)
        assert converted.dtype == numpy.dtype(gene_type)
        numpy.testing.assert_array_equal(converted, [[boundary, boundary]])
        numpy.testing.assert_array_equal(random_values, original_values)


@pytest.mark.parametrize("random_values", [numpy.array(-1), numpy.array([-1, 0], dtype=object),
                                            numpy.empty((0, 2), dtype=object)])
def test_unsigned_mutation_preserves_candidate_shape(random_values):
    ga_instance = make_ga()
    converted = ga_instance.mutation_change_gene_dtype_and_round(
        random_values, 0, numpy.uint8(0), False)
    assert numpy.shape(converted) == random_values.shape
    numpy.testing.assert_array_equal(converted, numpy.zeros(random_values.shape, dtype=numpy.uint8))


@pytest.mark.parametrize("gene_type", integer_types)
def test_fractional_offsets_are_added_before_integer_truncation(gene_type):
    ga_instance = make_ga(gene_type=gene_type)
    converted = ga_instance.mutation_change_gene_dtype_and_round(
        [-0.2, 0.2, 1.9], 0, gene_type(1), False)
    numpy.testing.assert_array_equal(converted, [0, 1, 2])


@pytest.mark.parametrize("gene_type", [int, float, object, numpy.float32])
@pytest.mark.parametrize("mutation_by_replacement", [False, True])
def test_other_gene_types_keep_values_outside_narrow_integer_limits(gene_type, mutation_by_replacement):
    ga_instance = make_ga(gene_type=gene_type)
    random_values = [-2**20, 2**20]
    converted = ga_instance.mutation_change_gene_dtype_and_round(
        random_values, 0, 0, mutation_by_replacement)
    numpy.testing.assert_array_equal(converted, random_values)


@pytest.mark.parametrize("gene_type", [int, object])
def test_python_integer_mutation_keeps_unbounded_scalar_values(gene_type):
    ga_instance = make_ga(gene_type=gene_type)
    converted = ga_instance.mutation_change_gene_dtype_and_round(1, 0, 2**80, False)
    assert converted == 2**80 + 1


@pytest.mark.parametrize("random_seed", range(20))
def test_reported_unsigned_population_runs_with_default_mutation(random_seed):
    ga_instance = make_ga(num_generations=30, num_parents_mating=4,
                          sol_per_pop=10, num_genes=6, random_seed=random_seed)
    ga_instance.run()
    assert ga_instance.generations_completed == 30
    assert ga_instance.population.dtype == numpy.dtype(numpy.uint8)


@pytest.mark.parametrize("mutation_type", ['random', 'adaptive'])
@pytest.mark.parametrize("use_probability", [False, True])
@pytest.mark.parametrize("gene_space", [None, [None, None, None], [[None, 1], [None, 1], [None, 1]]])
@pytest.mark.parametrize("mutation_by_replacement", [False, True])
def test_mutation_backends_share_integer_limits(mutation_type, use_probability, gene_space, mutation_by_replacement):
    parameters = dict(mutation_type=mutation_type, gene_space=gene_space,
                      mutation_by_replacement=mutation_by_replacement,
                      gene_type=[numpy.uint8, numpy.int8, numpy.uint64],
                      initial_population=[[0, 127, 2**64 - 1]] * 4,
                      random_mutation_min_val=[-2, 1, 1],
                      random_mutation_max_val=[-1, 2, 2],
                      crossover_type=None, keep_elitism=0, keep_parents=0)
    if use_probability:
        parameters['mutation_probability'] = [1.0, 1.0] if mutation_type == 'adaptive' else 1.0
    else:
        parameters['mutation_num_genes'] = [3, 3] if mutation_type == 'adaptive' else 3
    ga_instance = make_ga(**parameters)
    ga_instance.run()
    for solution in ga_instance.population:
        for gene_index, gene_type in enumerate([numpy.uint8, numpy.int8, numpy.uint64]):
            assert type(solution[gene_index]) is gene_type
        assert int(solution[0]) in [0, 1]
        if gene_space is None:
            numpy.testing.assert_array_equal(solution, [0, 1, 1] if mutation_by_replacement else [0, 127, 2**64 - 1])


def test_constraints_receive_clamped_mutation_candidates():
    candidate_batches = []

    def gene_constraint(solution, values):
        candidate_batches.append(values.copy())
        assert all(type(value) is numpy.uint8 and int(value) == 0 for value in values)
        return values

    ga_instance = make_ga(initial_population=[[0, 0, 0]] * 4,
                          gene_constraint=[gene_constraint] * 3,
                          random_mutation_min_val=-2, random_mutation_max_val=-1,
                          mutation_num_genes=3, keep_elitism=0, keep_parents=0)
    ga_instance.run()
    assert candidate_batches
    numpy.testing.assert_array_equal(ga_instance.population, 0)


@pytest.mark.parametrize("mutation_by_replacement", [False, True])
def test_duplicate_repair_uses_representable_mutation_candidates(mutation_by_replacement):
    ga_instance = make_ga(allow_duplicate_genes=False)
    solution, duplicate_indices, num_unsolved_duplicates = ga_instance.solve_duplicate_genes_randomly(
        numpy.array([0, 0, 255], dtype=numpy.uint8), -2, 2,
        mutation_by_replacement=mutation_by_replacement, gene_type=ga_instance.gene_type)
    assert not duplicate_indices
    assert num_unsolved_duplicates == 0
    assert solution.dtype == numpy.dtype(numpy.uint8)
    numpy.testing.assert_array_equal(numpy.sort(solution), [0, 1, 255])
