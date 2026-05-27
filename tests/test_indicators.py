import numpy
import pytest

from pygad.utils import indicators


# Two-objective example used for hand-derived expected values. Under
# the PyGAD-max convention the reference point must be strictly worse
# than every solution on every objective.
TWO_OBJECTIVE_FRONT = numpy.array([
    [-1.0, -8.0],
    [-3.0, -5.0],
    [-6.0, -2.0],
])
TWO_OBJECTIVE_REFERENCE = numpy.array([-10.0, -10.0])

# A second front used to verify IGD / GD assertions.
APPROXIMATION_FRONT = numpy.array([
    [-1.0, -8.0],
    [-4.0, -4.0],
    [-7.0, -1.0],
])


def test_hypervolume_two_d_matches_hand_computation():
    # Negate the points to switch to minimisation:
    #   ref = (10, 10), points = (1, 8), (3, 5), (6, 2)
    # The dominated area (computed by slicing the front from left to
    # right) is 4 + 15 + 32 = 51.
    expected_hv = 4.0 + 15.0 + 32.0
    hv = indicators.hypervolume(TWO_OBJECTIVE_FRONT, TWO_OBJECTIVE_REFERENCE)
    assert hv == pytest.approx(expected_hv, abs=1e-9)


def test_hypervolume_single_solution_equals_box_volume():
    # Single non-dominated point at (-2, -3) with reference (-10, -10).
    # Negated: point (2, 3), ref (10, 10). Box volume = (10-2) * (10-3)
    # = 8 * 7 = 56.
    fitness = numpy.array([[-2.0, -3.0]])
    reference = numpy.array([-10.0, -10.0])
    assert indicators.hypervolume(fitness, reference) == pytest.approx(56.0)


def test_hypervolume_drops_dominated_solutions():
    # The third row is dominated by both other rows. Adding it must
    # not change the hypervolume.
    extra = numpy.vstack([TWO_OBJECTIVE_FRONT, [[-4.0, -6.0]]])
    hv_clean = indicators.hypervolume(TWO_OBJECTIVE_FRONT, TWO_OBJECTIVE_REFERENCE)
    hv_with_dominated = indicators.hypervolume(extra, TWO_OBJECTIVE_REFERENCE)
    assert hv_clean == pytest.approx(hv_with_dominated, abs=1e-9)


def test_hypervolume_rejects_reference_point_inside_front():
    # The reference point must be strictly worse than every solution.
    fitness = numpy.array([[-1.0, -2.0], [-3.0, -1.0]])
    bad_reference = numpy.array([0.0, 0.0])
    with pytest.raises(ValueError, match="smaller than every solution"):
        indicators.hypervolume(fitness, bad_reference)


def test_hypervolume_three_d_axis_aligned_extremes():
    # Three axis-extreme solutions under PyGAD max:
    #   (-1, 0, 0), (0, -1, 0), (0, 0, -1).
    # Negated to minimisation:
    #   (1, 0, 0), (0, 1, 0), (0, 0, 1)  with reference (2, 2, 2).
    # Each solution dominates a 1 x 2 x 2 box of volume 4, but they
    # overlap. By inclusion-exclusion the union has volume
    # 4 + 4 + 4 - 2 - 2 - 2 + 1 = 7.
    fitness = numpy.array([
        [-1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0, -1.0],
    ])
    reference = numpy.array([-2.0, -2.0, -2.0])
    hv = indicators.hypervolume(fitness, reference)
    assert hv == pytest.approx(7.0, abs=1e-9)


def test_inverted_generational_distance_zero_when_approximation_matches_reference():
    igd = indicators.inverted_generational_distance(
        TWO_OBJECTIVE_FRONT, TWO_OBJECTIVE_FRONT)
    assert igd == pytest.approx(0.0, abs=1e-12)


def test_inverted_generational_distance_matches_hand_value():
    # For every row of TWO_OBJECTIVE_FRONT find the nearest row in
    # APPROXIMATION_FRONT under Euclidean distance, then average.
    # Hand check:
    #   ref (-1, -8) closest to approx (-1, -8) = distance 0
    #   ref (-3, -5) closest to approx (-4, -4) = sqrt(1 + 1) = sqrt(2)
    #   ref (-6, -2) closest to approx (-7, -1) = sqrt(1 + 1) = sqrt(2)
    expected = (0.0 + numpy.sqrt(2.0) + numpy.sqrt(2.0)) / 3.0
    igd = indicators.inverted_generational_distance(
        APPROXIMATION_FRONT, TWO_OBJECTIVE_FRONT)
    assert igd == pytest.approx(expected, abs=1e-12)


def test_generational_distance_matches_hand_value():
    # For every row of APPROXIMATION_FRONT find the nearest row in
    # TWO_OBJECTIVE_FRONT, then average.
    expected = (0.0 + numpy.sqrt(2.0) + numpy.sqrt(2.0)) / 3.0
    gd = indicators.generational_distance(
        APPROXIMATION_FRONT, TWO_OBJECTIVE_FRONT)
    assert gd == pytest.approx(expected, abs=1e-12)


def test_spacing_zero_for_equally_spaced_points():
    # Three colinear points equally spaced give nearest-neighbour
    # distances [1, 1, 1] (each endpoint sees its closest neighbour at
    # distance 1). The standard deviation is therefore 0.
    fitness = numpy.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])
    assert indicators.spacing(fitness) == pytest.approx(0.0, abs=1e-12)


def test_spacing_for_single_solution_returns_zero():
    # The metric is undefined for a single point; the implementation
    # short-circuits to 0.0 so the user does not have to special-case
    # it in their reporting code.
    fitness = numpy.array([[1.0, 2.0]])
    assert indicators.spacing(fitness) == 0.0


def test_hypervolume_random_four_dim_matches_pinned_value():
    # Pinned regression for a 4-objective case. The expected value
    # was computed once on this exact input array and stored here so
    # the test is fully self-contained (no external library calls).
    rng = numpy.random.default_rng(42)
    fitness_min = rng.uniform(0.0, 1.0, size=(20, 4))
    fitness_max = -fitness_min
    reference_max = numpy.array([-1.5, -1.5, -1.5, -1.5])
    expected_hv = 3.5205665111978677
    hv = indicators.hypervolume(fitness_max, reference_max)
    assert hv == pytest.approx(expected_hv, abs=1e-9)
