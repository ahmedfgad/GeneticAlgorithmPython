import pygad
import numpy

def fitness_func(ga_instance, solution, solution_idx):
    # simple fitness: sum of genes
    return numpy.sum(solution)

def test_integration_run():
    print("Starting Integration Test for Gene Structure Atomicity...")
    
    gene_structure = [2, 3, 1]
    # Total genes = 6
    
    # Define combinations to test
    mutation_types = ["random", "inversion", "scramble"]
    # excluded: "swap" (disabled)
    
    crossover_types = ["single_point", "two_points", "uniform", "scattered"]
    
    for mut_type in mutation_types:
        for cross_type in crossover_types:
            print(f"\nTesting Combination: Mutation='{mut_type}', Crossover='{cross_type}'")
            
            try:
                ga_instance = pygad.GA(num_generations=5,
                                       num_parents_mating=2,
                                       fitness_func=fitness_func,
                                       sol_per_pop=4,
                                       num_genes=sum(gene_structure),
                                       gene_structure=gene_structure,
                                       mutation_type=mut_type,
                                       crossover_type=cross_type,
                                       # Ensure mutation actually happens reasonably often
                                       mutation_probability=0.5, 
                                       crossover_probability=0.8,
                                       suppress_warnings=True)
                
                ga_instance.run()
                
                best_sol, best_fit, _ = ga_instance.best_solution()
                print(f"  > Success! Best Fitness: {best_fit}")
                
            except Exception as e:
                print(f"  > FAILED with error: {e}")
                raise e

    print("\nIntegration Test Completed Successfully.")

if __name__ == '__main__':
    test_integration_run()
