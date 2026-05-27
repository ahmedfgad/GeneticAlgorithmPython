import pygad
import numpy
import os
import matplotlib
# Use Agg backend for headless testing (no GUI needed)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global constants for testing
num_generations = 5
num_parents_mating = 4
sol_per_pop = 10
num_genes = 3
random_seed = 42

def fitness_func(ga_instance, solution, solution_idx):
    return numpy.sum(solution**2)

def fitness_func_multi(ga_instance, solution, solution_idx):
    return [numpy.sum(solution**2), numpy.sum(solution)]

def test_plot_fitness_parameters():
    """Test all parameters of plot_fitness()."""
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           random_seed=random_seed,
                           suppress_warnings=True
                           )
    ga_instance.run()
    
    # Test different plot types
    for p_type in ["plot", "scatter", "bar"]:
        fig = ga_instance.plot_fitness(plot_type=p_type, 
                                       title=f"Title {p_type}",
                                       xlabel="X", ylabel="Y",
                                       linewidth=2, font_size=12, color="blue")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    # Test multi-objective specific parameters
    ga_multi = pygad.GA(num_generations=2,
                        num_parents_mating=2,
                        fitness_func=fitness_func_multi,
                        sol_per_pop=5,
                        num_genes=3,
                        parent_selection_type="nsga2",
                        suppress_warnings=True)
    ga_multi.run()
    
    fig = ga_multi.plot_fitness(linewidth=[2, 4],
                                color=["blue", "green"],
                                label=["Obj A", "Obj B"])
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
    print("test_plot_fitness_parameters passed.")

def test_plot_new_solution_rate_parameters():
    """Test all parameters of plot_new_solution_rate() and its validation."""
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           random_seed=random_seed,
                           save_solutions=True,
                           suppress_warnings=True
                           )
    ga_instance.run()
    
    # Test different plot types and parameters
    for p_type in ["plot", "scatter", "bar"]:
        fig = ga_instance.plot_new_solution_rate(title=f"Rate {p_type}",
                                                plot_type=p_type,
                                                linewidth=2, color="purple")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    # Validation: Test error when save_solutions=False
    ga_instance_no_save = pygad.GA(num_generations=1,
                                   num_parents_mating=1,
                                   fitness_func=fitness_func,
                                   sol_per_pop=5,
                                   num_genes=2,
                                   save_solutions=False,
                                   suppress_warnings=True)
    ga_instance_no_save.run()
    try:
        ga_instance_no_save.plot_new_solution_rate()
    except RuntimeError:
        print("plot_new_solution_rate validation caught.")
        
    print("test_plot_new_solution_rate_parameters passed.")

def test_plot_genes_parameters():
    """Test all parameters of plot_genes()."""
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           random_seed=random_seed,
                           save_solutions=True,
                           save_best_solutions=True,
                           suppress_warnings=True
                           )
    ga_instance.run()
    
    # Test different graph types and parameters
    for g_type in ["plot", "boxplot", "histogram"]:
        fig = ga_instance.plot_genes(graph_type=g_type, fill_color="yellow", color="black")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
        
    # Test solutions="best"
    fig = ga_instance.plot_genes(solutions="best")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
    
    print("test_plot_genes_parameters passed.")

def test_plot_pareto_front_curve_parameters():
    """Test all parameters of plot_pareto_front_curve() and its validation."""
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func_multi,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           random_seed=random_seed,
                           parent_selection_type="nsga2",
                           suppress_warnings=True
                           )
    ga_instance.run()
    
    fig = ga_instance.plot_pareto_front_curve(title="Pareto",
                                             linewidth=4,
                                             label="Frontier",
                                             color="red",
                                             color_fitness="black",
                                             grid=False,
                                             alpha=0.5,
                                             marker="x")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
    
    # Validation: Test error for single-objective
    ga_instance_single = pygad.GA(num_generations=1,
                                  num_parents_mating=1,
                                  fitness_func=fitness_func,
                                  sol_per_pop=5,
                                  num_genes=2,
                                  suppress_warnings=True)
    ga_instance_single.run()
    try:
        ga_instance_single.plot_pareto_front_curve()
    except RuntimeError:
        print("plot_pareto_front_curve validation (multi-objective required) caught.")
        
    print("test_plot_pareto_front_curve_parameters passed.")

def test_visualize_save_dir():
    """Test save_dir parameter for all methods."""
    ga_instance = pygad.GA(num_generations=2,
                           num_parents_mating=2,
                           fitness_func=fitness_func,
                           sol_per_pop=5,
                           num_genes=2,
                           save_solutions=True,
                           suppress_warnings=True
                           )
    ga_instance.run()

    methods = [
        (ga_instance.plot_fitness, {}),
        (ga_instance.plot_new_solution_rate, {}),
        (ga_instance.plot_genes, {"graph_type": "plot"})
    ]

    for method, kwargs in methods:
        filename = f"test_{method.__name__}.png"
        if os.path.exists(filename): os.remove(filename)
        method(save_dir=filename, **kwargs)
        assert os.path.exists(filename)
        os.remove(filename)
    
    print("test_visualize_save_dir passed.")

def _build_three_objective_ga(num_generations=4, sol_per_pop=10, save_solutions=False):
    def fitness_func_three(ga_instance, solution, solution_idx):
        return [numpy.sum(solution ** 2),
                numpy.sum(solution),
                float(solution[0])]

    ga = pygad.GA(num_generations=num_generations,
                  num_parents_mating=4,
                  fitness_func=fitness_func_three,
                  sol_per_pop=sol_per_pop,
                  num_genes=3,
                  parent_selection_type="nsga2",
                  save_solutions=save_solutions,
                  random_seed=0,
                  suppress_warnings=True)
    ga.run()
    return ga


def test_plot_pareto_front_curve_three_objectives_returns_3d_figure():
    ga = _build_three_objective_ga()
    fig = ga.plot_pareto_front_curve()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_pareto_front_curve_rejects_four_or_more_objectives():
    def fitness_four(ga_instance, solution, solution_idx):
        return [float(solution[0]), float(solution[1]),
                float(solution[2]), float(solution[0]) + float(solution[1])]

    ga = pygad.GA(num_generations=2,
                  num_parents_mating=4,
                  fitness_func=fitness_four,
                  sol_per_pop=8,
                  num_genes=3,
                  parent_selection_type="nsga2",
                  random_seed=0,
                  suppress_warnings=True)
    ga.run()
    try:
        ga.plot_pareto_front_curve()
    except RuntimeError as exc:
        assert "2 or 3 objectives" in str(exc)
    else:
        raise AssertionError("plot_pareto_front_curve() should reject M >= 4")


def test_plot_pareto_front_pcp_returns_figure():
    ga = _build_three_objective_ga()
    fig = ga.plot_pareto_front_pcp()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_pareto_front_scatter_matrix_returns_figure():
    ga = _build_three_objective_ga()
    fig = ga.plot_pareto_front_scatter_matrix()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_pareto_front_heatmap_returns_figure_and_validates_sort_by():
    ga = _build_three_objective_ga()
    fig = ga.plot_pareto_front_heatmap()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
    try:
        ga.plot_pareto_front_heatmap(sort_by=99)
    except ValueError:
        pass
    else:
        raise AssertionError("plot_pareto_front_heatmap() should reject out-of-range sort_by")


def test_plot_fitness_band_requires_save_solutions():
    ga = pygad.GA(num_generations=3,
                  num_parents_mating=2,
                  fitness_func=fitness_func,
                  sol_per_pop=6,
                  num_genes=2,
                  random_seed=0,
                  suppress_warnings=True)
    ga.run()
    try:
        ga.plot_fitness_band()
    except RuntimeError as exc:
        assert "save_solutions" in str(exc)
    else:
        raise AssertionError("plot_fitness_band() should require save_solutions=True")


def test_plot_fitness_band_returns_figure_when_save_solutions_true():
    ga = pygad.GA(num_generations=3,
                  num_parents_mating=2,
                  fitness_func=fitness_func,
                  sol_per_pop=6,
                  num_genes=2,
                  save_solutions=True,
                  random_seed=0,
                  suppress_warnings=True)
    ga.run()
    fig = ga.plot_fitness_band()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_non_dominated_hypervolume_returns_figure():
    ga = _build_three_objective_ga(save_solutions=True)
    fig = ga.plot_non_dominated_hypervolume()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_non_dominated_hypervolume_requires_save_solutions():
    ga = _build_three_objective_ga(save_solutions=False)
    try:
        ga.plot_non_dominated_hypervolume()
    except RuntimeError as exc:
        assert "save_solutions" in str(exc)
    else:
        raise AssertionError("plot_non_dominated_hypervolume() should require save_solutions=True")


def test_plot_population_diversity_returns_figure():
    ga = pygad.GA(num_generations=4,
                  num_parents_mating=2,
                  fitness_func=fitness_func,
                  sol_per_pop=6,
                  num_genes=3,
                  save_solutions=True,
                  random_seed=0,
                  suppress_warnings=True)
    ga.run()
    fig = ga.plot_population_diversity()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_pareto_front_evolution_returns_figure():
    ga = _build_three_objective_ga(save_solutions=True)
    fig = ga.plot_pareto_front_evolution(every_k=2)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_pareto_front_evolution_rejects_non_positive_k():
    ga = _build_three_objective_ga(save_solutions=True)
    try:
        ga.plot_pareto_front_evolution(every_k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("plot_pareto_front_evolution() should reject every_k <= 0")


if __name__ == "__main__":
    test_plot_fitness_parameters()
    test_plot_new_solution_rate_parameters()
    test_plot_genes_parameters()
    test_plot_pareto_front_curve_parameters()
    test_visualize_save_dir()
    print("\nAll visualization tests passed!")
