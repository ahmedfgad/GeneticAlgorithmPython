"""Tests for configuration charts without executing the configured GA."""

import random
import subprocess
import sys

import numpy
import pytest

import pygad
from pygad.visualize.lifecycle import _describe_lifecycle

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as matplt


def fitness_func(ga_instance, solution, solution_idx):
    return float(numpy.sum(solution))


def create_ga_instance(**parameters):
    """Create a small GA, allowing each test to choose its configuration."""
    configuration = dict(num_generations=2, num_parents_mating=3,
                         sol_per_pop=8, num_genes=4,
                         fitness_func=fitness_func, random_seed=17,
                         suppress_warnings=True)
    configuration.update(parameters)
    return pygad.GA(**configuration)


def figure_text(fig):
    """Read the visible labels rather than inspecting rendering details."""
    return "\n".join(text.get_text().replace("\n", " ")
                     for axes in fig.axes for text in axes.texts)


def test_plot_lifecycle_before_run_has_no_execution_side_effects():
    calls = []

    def fitness_func_not_called(ga_instance, solution, solution_idx):
        calls.append("fitness")
        return 1.0

    def on_start(ga_instance):
        calls.append("on_start")

    def on_generation(ga_instance):
        calls.append("on_generation")

    ga_instance = create_ga_instance(fitness_func=fitness_func_not_called,
                                     on_start=on_start, on_generation=on_generation)
    original_population = ga_instance.population.copy()
    original_numpy_random_state = numpy.random.get_state()
    original_python_random_state = random.getstate()
    original_attribute_names = set(vars(ga_instance))
    fig = ga_instance.plot_lifecycle(show=False)
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        assert "Known after fitness evaluation" in figure_text(fig)
        assert calls == []
        assert ga_instance.generations_completed == 0
        assert ga_instance.last_generation_fitness is None
        assert set(vars(ga_instance)) == original_attribute_names
        numpy.testing.assert_array_equal(ga_instance.population, original_population)
        current_numpy_random_state = numpy.random.get_state()
        assert current_numpy_random_state[0] == original_numpy_random_state[0]
        numpy.testing.assert_array_equal(current_numpy_random_state[1], original_numpy_random_state[1])
        assert current_numpy_random_state[2:] == original_numpy_random_state[2:]
        assert random.getstate() == original_python_random_state
    finally:
        matplt.close(fig)


def test_lifecycle_callback_order_matches_execution_with_bypassed_operators():
    events = []

    def fitness_func_recorded(ga_instance, solution, solution_idx):
        events.append("fitness")
        return 1.0

    def on_start(ga_instance):
        events.append("on_start")

    def on_fitness(ga_instance, population_fitness):
        events.append("on_fitness")

    def on_parents(ga_instance, parents):
        events.append("on_parents")

    def on_crossover(ga_instance, offspring):
        events.append("on_crossover")

    def on_mutation(ga_instance, offspring):
        events.append("on_mutation")

    def on_generation(ga_instance):
        events.append("on_generation")
        return "stop"

    def on_stop(ga_instance, population_fitness):
        events.append("on_stop")

    ga_instance = create_ga_instance(fitness_func=fitness_func_recorded,
                                     crossover_type=None, mutation_type=None,
                                     keep_elitism=0, keep_parents=0,
                                     on_start=on_start, on_fitness=on_fitness,
                                     on_parents=on_parents, on_crossover=on_crossover,
                                     on_mutation=on_mutation,
                                     on_generation=on_generation, on_stop=on_stop)
    lifecycle = _describe_lifecycle(ga_instance)
    stages = {stage["id"]: stage for stage in lifecycle["stages"]}
    assert stages["crossover"]["kind"] == "bypass"
    assert stages["mutation"]["kind"] == "bypass"
    assert stages["generation_fitness"]["details"][0] == "fitness_func_recorded()"

    ga_instance.run()
    # Fitness executes once per solution in each evaluation; group
    # those calls to compare lifecycle stages to the observed order.
    observed_order = []
    for event in events:
        if event != "fitness" or not observed_order or observed_order[-1] != "fitness":
            observed_order.append(event)
    chart_order = []
    for stage in lifecycle["stages"]:
        if stage["kind"] == "callback":
            chart_order.append(stage["id"])
        elif stage["id"] in ("initial_fitness", "generation_fitness"):
            chart_order.append("fitness")
    assert chart_order == observed_order
    assert ga_instance.generations_completed == 1
    assert any(connection["source"] == "early_stop" and connection["target"] == "finalize"
               and connection["label"] == "Yes" for connection in lifecycle["connections"])


@pytest.mark.parametrize("num_generations", [0, 2])
def test_lifecycle_generation_loop_and_exit(num_generations):
    ga_instance = create_ga_instance(num_generations=num_generations)
    lifecycle = _describe_lifecycle(ga_instance)
    connections = lifecycle["connections"]
    assert any(connection["source"] == "generation_check" and connection["target"] == "finalize"
               and connection["label"] == "No" for connection in connections)
    assert any(connection["source"] == "generation_fitness" and connection["target"] == "generation_check"
               and connection["route"] == "repeat" for connection in connections)
    assert not any(connection["source"] == "generation_fitness" and connection["target"] == "finalize"
                   for connection in connections)
    assert not any(stage["kind"] == "callback" for stage in lifecycle["stages"])
    ga_instance.run()
    assert ga_instance.generations_completed == num_generations


@pytest.mark.parametrize("keep_elitism,keep_parents,retention_text,offspring_count", [
    (2, 0, "Keep 2 elite solution(s)", 6),
    (0, -1, "Keep all 3 selected parents", 5),
    (0, 2, "Keep 2 parent(s)", 6),
    (0, 0, "Keep no parents or elite solutions", 8),
])
def test_lifecycle_effective_population_retention(keep_elitism, keep_parents, retention_text, offspring_count):
    ga_instance = create_ga_instance(keep_elitism=keep_elitism, keep_parents=keep_parents)
    fig = ga_instance.plot_lifecycle(show=False)
    try:
        labels = figure_text(fig)
        assert retention_text in labels
        assert f"Add {offspring_count} offspring" in labels
        assert f"Offspring: ({offspring_count}, 4)" in labels
    finally:
        matplt.close(fig)


def test_lifecycle_custom_operators_compact_view():
    def select_custom_parents(fitness, num_parents, ga_instance):
        raise AssertionError("Drawing must not call the parent selector.")

    def create_custom_offspring(parents, offspring_size, ga_instance):
        raise AssertionError("Drawing must not call crossover.")

    def mutate_custom_offspring(offspring, ga_instance):
        raise AssertionError("Drawing must not call mutation.")

    ga_instance = create_ga_instance(parent_selection_type=select_custom_parents,
                                     crossover_type=create_custom_offspring,
                                     mutation_type=mutate_custom_offspring)
    fig = ga_instance.plot_lifecycle(show_parameters=False, show=False)
    try:
        labels = figure_text(fig)
        assert "select_custom_parents()" in labels
        assert "create_custom_offspring()" in labels
        assert "mutate_custom_offspring()" in labels
        assert "Configuration" not in labels
        assert "Offspring:" not in labels
    finally:
        matplt.close(fig)


@pytest.mark.parametrize("parent_selection_type", ["nsga2", "nsga3"])
def test_lifecycle_multi_objective_after_run(parent_selection_type):
    def fitness_func_multi(ga_instance, solution, solution_idx):
        return [float(numpy.sum(solution)), -float(numpy.sum(solution ** 2))]

    parameters = dict(fitness_func=fitness_func_multi, parent_selection_type=parent_selection_type)
    if parent_selection_type == "nsga3":
        parameters["nsga3_num_divisions"] = 2
    ga_instance = create_ga_instance(**parameters)
    ga_instance.run()
    original_evaluation_count = ga_instance.num_fitness_evaluations
    original_fitness = ga_instance.last_generation_fitness.copy()
    fig = ga_instance.plot_lifecycle(show=False)
    try:
        labels = figure_text(fig)
        assert "Population fitness: (8, 2)" in labels
        assert "Known after fitness evaluation" not in labels
        assert ("Prepare NSGA-III reference points" in labels) == (parent_selection_type == "nsga3")
        assert ga_instance.num_fitness_evaluations == original_evaluation_count
        numpy.testing.assert_array_equal(ga_instance.last_generation_fitness, original_fitness)
    finally:
        matplt.close(fig)


def test_lifecycle_mixed_genes_batching_constraints_and_stopping():
    def fitness_func_batch(ga_instance, solutions, solution_indices):
        raise AssertionError("Drawing must not evaluate a fitness batch.")

    def positive_gene_values(solution, values):
        return [value for value in values if value >= 0]

    ga_instance = create_ga_instance(fitness_func=fitness_func_batch, fitness_batch_size=3,
                                     gene_type=[int, [float, 2], int, float],
                                     gene_space=[range(1000), {"low": 0, "high": 10}, range(1000), None],
                                     gene_constraint=[positive_gene_values, None, None, None],
                                     parent_selection_type="tournament", K_tournament=4,
                                     mutation_type="adaptive", mutation_probability=[0.8, 0.2],
                                     parallel_processing=["thread", 2],
                                     stop_criteria=["reach_20", "saturate_3", "time_10", "evaluations_100"])
    fig = ga_instance.plot_lifecycle(show=False)
    try:
        labels = figure_text(fig)
        assert "Batch size: 3" in labels
        assert "Tournament size: 4" in labels
        assert "Probability per gene: [0.8, 0.2]" in labels
        assert "Evaluate offspring fitness to choose mutation" in labels
        assert "decimal places" in labels
        assert "1 constrained gene(s)" in labels
        assert "['thread', 2]" in labels
        for criterion in ["reach_20.0", "saturate_3.0", "time_10.0", "evaluations_100.0"]:
            assert criterion in labels
    finally:
        matplt.close(fig)


def test_lifecycle_polynomial_mutation_and_sbx_parameters():
    ga_instance = create_ga_instance(crossover_type="sbx", sbx_crossover_eta=15,
                                     mutation_type="polynomial", polynomial_mutation_eta=25)
    fig = ga_instance.plot_lifecycle(show=False)
    try:
        labels = figure_text(fig)
        assert "Distribution index: 15" in labels
        assert "Distribution index: 25" in labels
        # Polynomial mutation uses 1 / num_genes when no probability
        # is supplied, rather than mutation_num_genes.
        assert "Probability per gene: 0.25" in labels
        assert "Genes to mutate:" not in labels
    finally:
        matplt.close(fig)


@pytest.mark.parametrize("show_parameters,font_size", [(True, 11), (False, 16)])
def test_lifecycle_long_names_fit_in_the_chart(show_parameters, font_size):
    def fitness_func_with_long_name(ga_instance, solution, solution_idx):
        return 1.0

    # Wide glyphs expose clipping that a character-count limit misses.
    fitness_func_with_long_name.__name__ = "W" * 120
    ga_instance = create_ga_instance(fitness_func=fitness_func_with_long_name)
    fig = ga_instance.plot_lifecycle(title="Wide lifecycle " + "W" * 100,
                                     show_parameters=show_parameters,
                                     font_size=font_size, show=False)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for axes in fig.axes:
            for text in axes.texts:
                text_bounds = text.get_window_extent(renderer)
                assert fig.bbox.contains(*text_bounds.get_points()[0]), text.get_text()
                assert fig.bbox.contains(*text_bounds.get_points()[1]), text.get_text()
                if text.get_text().replace("\n", "").startswith("W" * 20):
                    # Handler labels must fit inside a card as well as
                    # inside the figure's outer boundary.
                    assert any(card.get_window_extent(renderer).contains(*text_bounds.get_points()[0])
                               and card.get_window_extent(renderer).contains(*text_bounds.get_points()[1])
                               for card in axes.patches), text.get_text()
    finally:
        matplt.close(fig)


@pytest.mark.parametrize("extension", ["svg", "png", "pdf"])
def test_lifecycle_export_and_display_control(tmp_path, monkeypatch, extension):
    shown_figures = []
    monkeypatch.setattr(matplt, "show", lambda: shown_figures.append(True))
    output_path = tmp_path / ("lifecycle." + extension)
    ga_instance = create_ga_instance()
    fig = ga_instance.plot_lifecycle(title="Scheduling $problem$", save_dir=output_path, show=False)
    try:
        assert shown_figures == []
        assert output_path.stat().st_size > 1000
        assert "Scheduling $problem$" in figure_text(fig)
    finally:
        matplt.close(fig)
    fig = ga_instance.plot_lifecycle()
    assert shown_figures == [True]
    matplt.close(fig)


@pytest.mark.parametrize("parameters,error", [
    ({"title": None}, TypeError),
    ({"font_size": "large"}, TypeError),
    ({"font_size": True}, TypeError),
    ({"font_size": 0}, ValueError),
    ({"font_size": -1}, ValueError),
    ({"font_size": numpy.nan}, ValueError),
    ({"font_size": numpy.inf}, ValueError),
    ({"show_parameters": "yes"}, TypeError),
    ({"show": None}, TypeError),
])
def test_lifecycle_parameter_validation(parameters, error):
    with pytest.raises(error):
        create_ga_instance().plot_lifecycle(**parameters)


def test_lifecycle_optional_matplotlib_import(monkeypatch):
    # Importing PyGAD and building its lifecycle description stay usable
    # without the plotting extra installed.
    result = subprocess.run([sys.executable, "-c",
                             "import sys; import pygad; "
                             "from pygad.visualize.lifecycle import _describe_lifecycle; "
                             "assert 'matplotlib.pyplot' not in sys.modules"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    def missing_matplotlib():
        raise ImportError("matplotlib is unavailable")

    monkeypatch.setattr(pygad.visualize.plot, "get_matplotlib", missing_matplotlib)
    with pytest.raises(ImportError, match=r"pip install pygad\[visualize\]"):
        create_ga_instance().plot_lifecycle(show=False)
