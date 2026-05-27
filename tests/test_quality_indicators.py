import numpy
import pytest

from pygad.utils import quality_indicators


# 2-objective fixture used for hand-derived expected values. Under
# PyGAD-max the reference must be strictly worse than every solution.
TWO_OBJECTIVE_FRONT = numpy.array([
    [-1.0, -8.0],
    [-3.0, -5.0],
    [-6.0, -2.0],
])
TWO_OBJECTIVE_REFERENCE = numpy.array([-10.0, -10.0])

# Second front used by the IGD / GD tests.
APPROXIMATION_FRONT = numpy.array([
    [-1.0, -8.0],
    [-4.0, -4.0],
    [-7.0, -1.0],
])


def test_hypervolume_two_d_matches_hand_computation():
    # In min space: ref (10, 10), points (1, 8), (3, 5), (6, 2).
    # Sliced area from left to right: 4 + 15 + 32 = 51.
    expected_hv = 4.0 + 15.0 + 32.0
    hv = quality_indicators.hypervolume(TWO_OBJECTIVE_FRONT, TWO_OBJECTIVE_REFERENCE)
    assert hv == pytest.approx(expected_hv, abs=1e-9)


def test_hypervolume_single_solution_equals_box_volume():
    # Point (-2, -3), ref (-10, -10). In min: box = 8 * 7 = 56.
    fitness = numpy.array([[-2.0, -3.0]])
    reference = numpy.array([-10.0, -10.0])
    assert quality_indicators.hypervolume(fitness, reference) == pytest.approx(56.0)


def test_hypervolume_drops_dominated_solutions():
    # The added row is dominated, so HV should not change.
    extra = numpy.vstack([TWO_OBJECTIVE_FRONT, [[-4.0, -6.0]]])
    hv_clean = quality_indicators.hypervolume(TWO_OBJECTIVE_FRONT, TWO_OBJECTIVE_REFERENCE)
    hv_with_dominated = quality_indicators.hypervolume(extra, TWO_OBJECTIVE_REFERENCE)
    assert hv_clean == pytest.approx(hv_with_dominated, abs=1e-9)


def test_hypervolume_rejects_reference_point_inside_front():
    # The reference point must be strictly worse than every solution.
    fitness = numpy.array([[-1.0, -2.0], [-3.0, -1.0]])
    bad_reference = numpy.array([0.0, 0.0])
    with pytest.raises(ValueError, match="smaller than every solution"):
        quality_indicators.hypervolume(fitness, bad_reference)


def test_hypervolume_three_d_axis_aligned_extremes():
    # Three axis-extreme points, ref (-2,-2,-2). In min: each point
    # dominates a 1x2x2 box, by inclusion-exclusion union = 7.
    fitness = numpy.array([
        [-1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0, -1.0],
    ])
    reference = numpy.array([-2.0, -2.0, -2.0])
    hv = quality_indicators.hypervolume(fitness, reference)
    assert hv == pytest.approx(7.0, abs=1e-9)


def test_inverted_generational_distance_zero_when_approximation_matches_reference():
    igd = quality_indicators.inverted_generational_distance(
        TWO_OBJECTIVE_FRONT, TWO_OBJECTIVE_FRONT)
    assert igd == pytest.approx(0.0, abs=1e-12)


def test_inverted_generational_distance_matches_hand_value():
    # Nearest approx per ref:
    #   (-1,-8) -> (-1,-8) = 0
    #   (-3,-5) -> (-4,-4) = sqrt(2)
    #   (-6,-2) -> (-7,-1) = sqrt(2)
    expected = (0.0 + numpy.sqrt(2.0) + numpy.sqrt(2.0)) / 3.0
    igd = quality_indicators.inverted_generational_distance(
        APPROXIMATION_FRONT, TWO_OBJECTIVE_FRONT)
    assert igd == pytest.approx(expected, abs=1e-12)


def test_generational_distance_matches_hand_value():
    # Symmetric of the IGD test: same pairings, same average.
    expected = (0.0 + numpy.sqrt(2.0) + numpy.sqrt(2.0)) / 3.0
    gd = quality_indicators.generational_distance(
        APPROXIMATION_FRONT, TWO_OBJECTIVE_FRONT)
    assert gd == pytest.approx(expected, abs=1e-12)


def test_spacing_zero_for_equally_spaced_points():
    # Three colinear, equally spaced points -> nearest-neighbour
    # distances are all 1 -> std = 0.
    fitness = numpy.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])
    assert quality_indicators.spacing(fitness) == pytest.approx(0.0, abs=1e-12)


def test_spacing_for_single_solution_returns_zero():
    # Undefined for one point; we return 0.0 to keep callers simple.
    fitness = numpy.array([[1.0, 2.0]])
    assert quality_indicators.spacing(fitness) == 0.0


def test_hypervolume_random_four_dim_matches_pinned_value():
    # 4-objective regression. Expected value was computed once on
    # this exact array and pinned so the test is self-contained.
    rng = numpy.random.default_rng(42)
    fitness_min = rng.uniform(0.0, 1.0, size=(20, 4))
    fitness_max = -fitness_min
    reference_max = numpy.array([-1.5, -1.5, -1.5, -1.5])
    expected_hv = 3.5205665111978677
    hv = quality_indicators.hypervolume(fitness_max, reference_max)
    assert hv == pytest.approx(expected_hv, abs=1e-9)
