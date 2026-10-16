"""
Tests for how SBX crossover and polynomial mutation convert their real
valued results back into integer and finite gene values.

Both operators compute a decimal number between the gene bounds. The
decimal must then become a value the gene accepts. These tests check
that this conversion picks the nearest accepted value. Chopping the
fraction instead makes the highest value unreachable and pulls every
gene toward zero.
"""

import numpy

import pygad


# Number of offspring rows used to measure each distribution. It is large
# enough to make the shares below stable for a fixed seed.
NUM_TRIALS = 3000


def _zero_fitness(ga, solution, solution_idx):
    return 0.0


def _make_one_gene_ga(gene_space, gene_type=int, **operator_settings):
    """Build a GA with a single gene so each operator can be called directly."""
    return pygad.GA(num_generations=1,
                    num_parents_mating=2,
                    fitness_func=_zero_fitness,
                    sol_per_pop=4,
                    num_genes=1,
                    gene_space=[gene_space],
                    gene_type=gene_type,
                    mutation_num_genes=1,
                    random_seed=1,
                    suppress_warnings=True,
                    **operator_settings)


def _mutate_from(ga, start_value):
    """Apply polynomial mutation to many copies of one value and return the results."""
    offspring = numpy.full((NUM_TRIALS, 1), start_value)
    return ga.polynomial_mutation(offspring)[:, 0].astype(float)


def _share(values, condition):
    return float(numpy.mean(condition(values)))


def test_polynomial_mutation_reaches_the_highest_integer_value():
    # A gene with the values 0 to 4 starting at 3 must sometimes move to 4.
    ga = _make_one_gene_ga({"low": 0, "high": 5, "step": 1},
                           mutation_type="polynomial",
                           polynomial_mutation_eta=5)
    values = _mutate_from(ga, 3)
    assert _share(values, lambda v: v == 4) > 0.05


def test_polynomial_mutation_reaches_the_last_value_of_a_list_space():
    # The same rule applies to a gene space given as a list.
    ga = _make_one_gene_ga([0, 1, 2, 3, 4],
                           mutation_type="polynomial",
                           polynomial_mutation_eta=5)
    values = _mutate_from(ga, 3)
    assert _share(values, lambda v: v == 4) > 0.05


def test_polynomial_mutation_reaches_the_highest_value_of_a_continuous_integer_space():
    # A continuous gene space used with an integer gene type. The 'high'
    # value is excluded, so the allowed integers are 0 to 3.
    ga = _make_one_gene_ga({"low": 0, "high": 4},
                           mutation_type="polynomial",
                           polynomial_mutation_eta=5)
    values = _mutate_from(ga, 2)
    assert _share(values, lambda v: v == 3) > 0.05


def test_polynomial_mutation_moves_up_and_down_equally_often():
    # From the middle value, moving down and moving up must be about
    # equally likely. Chopping the fraction makes moving down far more
    # likely than moving up.
    ga = _make_one_gene_ga({"low": 0, "high": 5, "step": 1},
                           mutation_type="polynomial",
                           polynomial_mutation_eta=5)
    values = _mutate_from(ga, 2)
    moved_down = _share(values, lambda v: v < 2)
    moved_up = _share(values, lambda v: v > 2)
    assert abs(moved_down - moved_up) < 0.05


def test_polynomial_mutation_is_balanced_on_a_negative_range():
    # Chopping the fraction pulls values toward zero, which is upward for
    # negative numbers. Starting at -2 in the range -4 to 0 checks that.
    ga = _make_one_gene_ga({"low": -4, "high": 1, "step": 1},
                           mutation_type="polynomial",
                           polynomial_mutation_eta=5)
    values = _mutate_from(ga, -2)
    moved_down = _share(values, lambda v: v < -2)
    moved_up = _share(values, lambda v: v > -2)
    assert abs(moved_down - moved_up) < 0.05


def test_polynomial_mutation_is_balanced_on_a_stepped_space():
    # With the values 0, 2, 4, 6, 8 the decimal result must go to the
    # nearest allowed value. Rounding it to an integer first would send
    # results such as 5.2 to 4 instead of 6.
    ga = _make_one_gene_ga({"low": 0, "high": 9, "step": 2},
                           mutation_type="polynomial",
                           polynomial_mutation_eta=5)
    values = _mutate_from(ga, 4)
    moved_down = _share(values, lambda v: v < 4)
    moved_up = _share(values, lambda v: v > 4)
    assert abs(moved_down - moved_up) < 0.05


def test_polynomial_mutation_keeps_values_inside_the_gene_space():
    # Rounding must never produce a value outside the allowed values.
    spaces = [{"low": 0, "high": 5, "step": 1},
              {"low": 0, "high": 9, "step": 2},
              {"low": -4, "high": 1, "step": 1},
              [16, 32, 64]]
    for space in spaces:
        ga = _make_one_gene_ga(space,
                               mutation_type="polynomial",
                               polynomial_mutation_eta=5)
        low, high = ga.get_bounded_operator_gene_range(0)
        for start_value in (low, high):
            for value in _mutate_from(ga, start_value):
                assert ga.is_gene_value_in_space(0, value, value), (space, value)


def test_sbx_crossover_can_keep_the_highest_parent_value():
    # With parents 1 and 4 in the range 0 to 4, a child must sometimes
    # keep the value 4. Chopping the fraction turns it into 3 every time.
    ga = _make_one_gene_ga({"low": 0, "high": 5, "step": 1},
                           crossover_type="sbx",
                           sbx_crossover_eta=5)
    parents = numpy.array([[1], [4]])
    children = ga.sbx_crossover(parents, (NUM_TRIALS, 1))[:, 0].astype(float)
    assert _share(children, lambda v: v == 4) > 0.3
    assert _share(children, lambda v: v == 1) > 0.3


def test_float_genes_are_not_rounded():
    # Float genes keep their decimal values.
    ga = _make_one_gene_ga({"low": 0.0, "high": 4.0},
                           gene_type=float,
                           mutation_type="polynomial",
                           polynomial_mutation_eta=5)
    values = _mutate_from(ga, 2.0)
    assert _share(values, lambda v: v != numpy.round(v)) > 0.9
