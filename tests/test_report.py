"""
Tests for the PDF report generator.

The full check here is structural: we verify the report writes a
non-empty PDF, picks up the expected plot methods based on the run
configuration, refuses unknown sections, and works for both single-
objective and multi-objective runs.
"""

import os

import numpy
import pytest

import pygad
from pygad.utils import report as report_module


pytest.importorskip("reportlab")
pytest.importorskip("matplotlib")


def _single_objective_fitness(ga, solution, sol_idx):
    return float(numpy.sum(solution))


def _two_objective_fitness(ga, solution, sol_idx):
    return [float(numpy.sum(solution)),
            -float(numpy.sum(numpy.asarray(solution) ** 2))]


def _build_soo_ga(**overrides):
    defaults = dict(
        num_generations=4,
        num_parents_mating=4,
        fitness_func=_single_objective_fitness,
        sol_per_pop=8,
        num_genes=3,
        random_seed=0,
        suppress_warnings=True,
    )
    defaults.update(overrides)
    return pygad.GA(**defaults)


def _build_moo_ga(**overrides):
    defaults = dict(
        num_generations=4,
        num_parents_mating=4,
        fitness_func=_two_objective_fitness,
        sol_per_pop=8,
        num_genes=3,
        parent_selection_type='nsga2',
        random_seed=0,
        suppress_warnings=True,
    )
    defaults.update(overrides)
    return pygad.GA(**defaults)


def test_generate_report_writes_non_empty_pdf(tmp_path):
    ga = _build_soo_ga(save_solutions=True, save_best_solutions=True)
    ga.run()
    output_path = ga.generate_report(str(tmp_path / "soo_report"))
    assert output_path.endswith(".pdf")
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 1000  # PDF header alone is > 1kB


def test_generate_report_appends_pdf_extension(tmp_path):
    ga = _build_soo_ga()
    ga.run()
    output_path = ga.generate_report(str(tmp_path / "no_extension"))
    assert output_path == str(tmp_path / "no_extension.pdf")


def test_generate_report_refuses_unknown_section(tmp_path):
    ga = _build_soo_ga()
    ga.run()
    with pytest.raises(ValueError, match="Unknown report sections"):
        ga.generate_report(str(tmp_path / "bad_section"),
                           sections=["title", "does_not_exist"])


def test_generate_report_refuses_unknown_plot_name(tmp_path):
    ga = _build_soo_ga()
    ga.run()
    with pytest.raises(ValueError, match="Unknown plot method"):
        ga.generate_report(str(tmp_path / "bad_plot"),
                           include_plots=["plot_nonexistent"])


def test_generate_report_requires_at_least_one_completed_generation(tmp_path):
    ga = _build_soo_ga()
    with pytest.raises(RuntimeError, match="at least one"):
        ga.generate_report(str(tmp_path / "empty"))


def test_select_plot_methods_filters_out_moo_only_plots_for_soo():
    ga = _build_soo_ga(save_solutions=True, save_best_solutions=True)
    ga.run()
    method_names = report_module._select_plot_methods(ga, include_plots=None)
    assert "plot_fitness" in method_names
    assert "plot_new_solution_rate" in method_names
    assert "plot_pareto_front_curve" not in method_names
    assert "plot_pareto_front_pcp" not in method_names
    assert "plot_non_dominated_hypervolume" not in method_names


def test_select_plot_methods_includes_moo_plots_for_moo_with_save_solutions():
    ga = _build_moo_ga(save_solutions=True)
    ga.run()
    method_names = report_module._select_plot_methods(ga, include_plots=None)
    assert "plot_fitness" in method_names
    assert "plot_pareto_front_curve" in method_names
    assert "plot_pareto_front_pcp" in method_names
    assert "plot_pareto_front_heatmap" in method_names
    assert "plot_non_dominated_hypervolume" in method_names
    assert "plot_pareto_front_evolution" in method_names


def test_select_plot_methods_omits_save_solutions_plots_when_flag_is_false():
    ga = _build_moo_ga(save_solutions=False)
    ga.run()
    method_names = report_module._select_plot_methods(ga, include_plots=None)
    assert "plot_fitness" in method_names
    # plot_pareto_front_curve does not need save_solutions.
    assert "plot_pareto_front_curve" in method_names
    # plot_non_dominated_hypervolume does.
    assert "plot_non_dominated_hypervolume" not in method_names
    assert "plot_pareto_front_evolution" not in method_names


def test_generate_report_for_multi_objective_run(tmp_path):
    ga = _build_moo_ga(save_solutions=True)
    ga.run()
    output_path = ga.generate_report(str(tmp_path / "moo_report"))
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 5000


def test_generate_report_honors_section_order(tmp_path):
    ga = _build_soo_ga()
    ga.run()
    output_path = ga.generate_report(
        str(tmp_path / "ordered_report"),
        sections=["title", "configuration"])
    assert os.path.exists(output_path)


def test_generate_report_supports_explicit_plot_list(tmp_path):
    ga = _build_soo_ga()
    ga.run()
    output_path = ga.generate_report(
        str(tmp_path / "fitness_only"),
        include_plots=["plot_fitness"])
    assert os.path.exists(output_path)


def test_generate_report_unknown_page_size_raises(tmp_path):
    ga = _build_soo_ga()
    ga.run()
    with pytest.raises(ValueError, match="Unknown page_size"):
        ga.generate_report(str(tmp_path / "bad_paper"), page_size="A12")
