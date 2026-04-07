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

if __name__ == "__main__":
    test_plot_fitness_parameters()
    test_plot_new_solution_rate_parameters()
    test_plot_genes_parameters()
    test_plot_pareto_front_curve_parameters()
    test_visualize_save_dir()
    print("\nAll visualization tests passed!")
