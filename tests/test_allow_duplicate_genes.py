import pygad
import random
import numpy

num_generations = 1

initial_population = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

def number_duplicate_genes(gene_space=None,
                           gene_type=float,
                           num_genes=10,
                           mutation_by_replacement=False,
                           random_mutation_min_val=-1,
                           random_mutation_max_val=1,
                           init_range_low=-4,
                           init_range_high=4,
                           random_seed=123,
                           initial_population=None,
                           parent_selection_type='sss',
                           multi_objective=False):

    def fitness_func_no_batch_single(ga, solution, idx):
        return random.random()

    def fitness_func_no_batch_multi(ga, solution, idx):
        return [random.random(), random.random()]

    if multi_objective == True:
        fitness_func = fitness_func_no_batch_multi
    else:
        fitness_func = fitness_func_no_batch_single

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=10,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           gene_type=gene_type,
                           initial_population=initial_population,
                           parent_selection_type=parent_selection_type,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           random_mutation_min_val=random_mutation_min_val,
                           random_mutation_max_val=random_mutation_max_val,
                           allow_duplicate_genes=False,
                           mutation_by_replacement=mutation_by_replacement,
                           random_seed=random_seed,
                           save_solutions=True,
                           suppress_warnings=True)

    ga_instance.run()

    num_duplicates = 0
    for solution in ga_instance.solutions:
        num = len(solution) - len(set(solution))
        if num != 0:
            print(solution)
        num_duplicates += num

    print(f"Number of duplicates is {num_duplicates}.")
    return num_duplicates

#### Single-Objective
def test_number_duplicates_default():
    num_duplicates = number_duplicate_genes()
    
    assert num_duplicates == 0

def test_number_duplicates_default_initial_population():
    num_duplicates = number_duplicate_genes(initial_population=initial_population)
    
    assert num_duplicates == 0

def test_number_duplicates_float_gene_type():
    num_genes = 10
    num_duplicates = number_duplicate_genes(gene_type=float,
                                            num_genes=num_genes,
                                            init_range_low=0,
                                            init_range_high=1,
                                            random_mutation_min_val=0,
                                            random_mutation_max_val=1)

    assert num_duplicates == 0

def test_number_duplicates_float_gene_type_initial_population():
    num_genes = 10
    num_duplicates = number_duplicate_genes(gene_type=float,
                                            num_genes=num_genes,
                                            init_range_low=0,
                                            init_range_high=1,
                                            initial_population=initial_population,
                                            random_mutation_min_val=0,
                                            random_mutation_max_val=1)

    assert num_duplicates == 0

def test_number_duplicates_int_gene_type():
    num_genes = 10
    init_range_low = 0
    init_range_high = init_range_low + num_genes
    random_mutation_min_val = 0
    random_mutation_max_val = random_mutation_min_val + num_genes
    num_duplicates = number_duplicate_genes(gene_type=int,
                                            mutation_by_replacement=False,
                                            num_genes=num_genes,
                                            init_range_low=init_range_low,
                                            init_range_high=init_range_high,
                                            random_mutation_min_val=random_mutation_min_val,
                                            random_mutation_max_val=random_mutation_max_val)

    assert num_duplicates == 0

def test_number_duplicates_int_gene_type_initial_population():
    num_genes = 10
    init_range_low = 0
    init_range_high = init_range_low + num_genes
    random_mutation_min_val = 0
    random_mutation_max_val = random_mutation_min_val + num_genes
    num_duplicates = number_duplicate_genes(gene_type=int,
                                            mutation_by_replacement=False,
                                            num_genes=num_genes,
                                            init_range_low=init_range_low,
                                            init_range_high=init_range_high,
                                            initial_population=initial_population,
                                            random_mutation_min_val=random_mutation_min_val,
                                            random_mutation_max_val=random_mutation_max_val)

    assert num_duplicates == 0

def test_number_duplicates_int_gene_type_replacement():
    num_genes = 10
    init_range_low = 0
    init_range_high = init_range_low + num_genes
    random_mutation_min_val = 0
    random_mutation_max_val = random_mutation_min_val + num_genes
    num_duplicates = number_duplicate_genes(gene_type=int,
                                            mutation_by_replacement=True,
                                            num_genes=num_genes,
                                            init_range_low=init_range_low,
                                            init_range_high=init_range_high,
                                            random_mutation_min_val=random_mutation_min_val,
                                            random_mutation_max_val=random_mutation_max_val)

    assert num_duplicates == 0

def test_number_duplicates_int_gene_type_replacement_initial_population():
    num_genes = 10
    init_range_low = 0
    init_range_high = init_range_low + num_genes
    random_mutation_min_val = 0
    random_mutation_max_val = random_mutation_min_val + num_genes
    num_duplicates = number_duplicate_genes(gene_type=int,
                                            mutation_by_replacement=True,
                                            num_genes=num_genes,
                                            init_range_low=init_range_low,
                                            init_range_high=init_range_high,
                                            initial_population=initial_population,
                                            random_mutation_min_val=random_mutation_min_val,
                                            random_mutation_max_val=random_mutation_max_val)

    assert num_duplicates == 0

def test_number_duplicates_single_gene_space():
    num_duplicates = number_duplicate_genes(gene_space=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                            num_genes=10)

    assert num_duplicates == 0

def test_number_duplicates_single_gene_space_initial_population():
    num_duplicates = number_duplicate_genes(gene_space=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                            num_genes=10,
                                            initial_population=initial_population)

    assert num_duplicates == 0

def test_number_duplicates_single_range_gene_space():
    num_genes = 10
    num_duplicates = number_duplicate_genes(gene_space=range(num_genes),
                                            num_genes=num_genes)

    assert num_duplicates == 0

def test_number_duplicates_single_range_gene_space_initial_population():
    num_genes = 10
    num_duplicates = number_duplicate_genes(gene_space=range(num_genes),
                                            num_genes=num_genes,
                                            initial_population=initial_population)

    assert num_duplicates == 0

def test_number_duplicates_single_numpy_range_gene_space():
    num_genes = 10
    num_duplicates = number_duplicate_genes(gene_space=numpy.arange(num_genes),
                                            num_genes=num_genes)

    assert num_duplicates == 0

def test_number_duplicates_single_numpy_range_gene_space_initial_population():
    num_genes = 10
    num_duplicates = number_duplicate_genes(gene_space=numpy.arange(num_genes),
                                            num_genes=num_genes,
                                            initial_population=initial_population)

    assert num_duplicates == 0

def test_number_duplicates_nested_gene_space():
    num_duplicates = number_duplicate_genes(gene_space=[[0, 1], 
                                                        [1, 2], 
                                                        [2, 3],
                                                        [3, 4],
                                                        [4, 5],
                                                        [5, 6],
                                                        [6, 7],
                                                        [7, 8],
                                                        [8, 9],
                                                        [9, 10]],
                                            gene_type=int,
                                            num_genes=10)

    assert num_duplicates == 0

def test_number_duplicates_nested_gene_space_initial_population():
    num_duplicates = number_duplicate_genes(gene_space=[[0, 1], 
                                                        [1, 2], 
                                                        [2, 3],
                                                        [3, 4],
                                                        [4, 5],
                                                        [5, 6],
                                                        [6, 7],
                                                        [7, 8],
                                                        [8, 9],
                                                        [9, 10]],
                                            gene_type=int,
                                            num_genes=10,
                                            initial_population=initial_population)

    assert num_duplicates == 0


# def test_number_duplicates_nested_gene_space_nested_gene_type():
    """
    This example causes duplicate genes that can only be solved by changing the values of a chain of genes.
    Let's explain it using this solution: [0, 2, 3, 4, 5, 6, 6, 7, 8, 9]
    It has 2 genes with the value 6 at indices 5 and 6.
    According to the gene space, none of these genes can has a different value that solves the duplicates.
        -If the value of the gene at index 5 is changed from 6 to 5, then it causes another duplicate with the gene at index 4.
        -If the value of the gene at index 6 is changed from 6 to 7, then it causes another duplicate with the gene at index 7.
    The solution is to change a chain of genes that make a room to solve the duplicates between the 2 genes.
        1) Change the second gene from 2 to 1.
        2) Change the third gene from 3 to 2.
        3) Change the fourth gene from 4 to 3.
        4) Change the fifth gene from 5 to 4.
        5) Change the sixth gene from 6 to 5. This solves the duplicates.
    But this is NOT SUPPORTED yet.
    We support changing only a single gene that makes a room to solve the duplicates.

    Let's explain it using this solution: [1, 2, 2, 4, 5, 6, 6, 7, 8, 9]
    It has 2 genes with the value 2 at indices 1 and 2.
    This is how the duplicates are solved:
        1) Change the first gene from 1 to 0.
        2) Change the second gene from 2 to 1. This solves the duplicates.
    The result is [0, 1, 2, 4, 5, 6, 6, 7, 8, 9]
    """
    # num_duplicates = number_duplicate_genes(gene_space=[[0, 1], 
    #                                                     [1, 2], 
    #                                                     [2, 3],
    #                                                     [3, 4],
    #                                                     [4, 5],
    #                                                     [5, 6],
    #                                                     [6, 7],
    #                                                     [7, 8],
    #                                                     [8, 9],
    #                                                     [9, 10]],
    #                                         gene_type=[int, int, int, int, int, int, int, int, int, int],
    #                                         num_genes=10)

    # assert num_duplicates == 0

def test_number_duplicates_nested_gene_space_nested_gene_type_initial_population():
    num_duplicates = number_duplicate_genes(gene_space=[[0, 1], 
                                                        [1, 2], 
                                                        [2, 3],
                                                        [3, 4],
                                                        [4, 5],
                                                        [5, 6],
                                                        [6, 7],
                                                        [7, 8],
                                                        [8, 9],
                                                        [9, 10]],
                                            gene_type=[int, int, int, int, int, int, int, int, int, int],
                                            num_genes=10,
                                            initial_population=initial_population)

    assert num_duplicates == 0

#### Multi-Objective
def test_number_duplicates_default_multi_objective():
    num_duplicates = number_duplicate_genes()
    
    assert num_duplicates == 0

def test_number_duplicates_default_initial_population_multi_objective():
    num_duplicates = number_duplicate_genes(initial_population=initial_population)
    
    assert num_duplicates == 0

def test_number_duplicates_float_gene_type_multi_objective():
    num_genes = 10
    num_duplicates = number_duplicate_genes(gene_type=float,
                                            num_genes=num_genes,
                                            init_range_low=0,
                                            init_range_high=1,
                                            random_mutation_min_val=0,
                                            random_mutation_max_val=1)

    assert num_duplicates == 0

def test_number_duplicates_float_gene_type_initial_population_multi_objective():
    num_genes = 10
    num_duplicates = number_duplicate_genes(gene_type=float,
                                            num_genes=num_genes,
                                            init_range_low=0,
                                            init_range_high=1,
                                            initial_population=initial_population,
                                            random_mutation_min_val=0,
                                            random_mutation_max_val=1)

    assert num_duplicates == 0

def test_number_duplicates_int_gene_type_multi_objective():
    num_genes = 10
    init_range_low = 0
    init_range_high = init_range_low + num_genes
    random_mutation_min_val = 0
    random_mutation_max_val = random_mutation_min_val + num_genes
    num_duplicates = number_duplicate_genes(gene_type=int,
                                            mutation_by_replacement=False,
                                            num_genes=num_genes,
                                            init_range_low=init_range_low,
                                            init_range_high=init_range_high,
                                            random_mutation_min_val=random_mutation_min_val,
                                            random_mutation_max_val=random_mutation_max_val)

    assert num_duplicates == 0

def test_number_duplicates_int_gene_type_initial_population_multi_objective():
    num_genes = 10
    init_range_low = 0
    init_range_high = init_range_low + num_genes
    random_mutation_min_val = 0
    random_mutation_max_val = random_mutation_min_val + num_genes
    num_duplicates = number_duplicate_genes(gene_type=int,
                                            mutation_by_replacement=False,
                                            num_genes=num_genes,
                                            init_range_low=init_range_low,
                                            init_range_high=init_range_high,
                                            initial_population=initial_population,
                                            random_mutation_min_val=random_mutation_min_val,
                                            random_mutation_max_val=random_mutation_max_val)

    assert num_duplicates == 0

def test_number_duplicates_int_gene_type_replacement_multi_objective():
    num_genes = 10
    init_range_low = 0
    init_range_high = init_range_low + num_genes
    random_mutation_min_val = 0
    random_mutation_max_val = random_mutation_min_val + num_genes
    num_duplicates = number_duplicate_genes(gene_type=int,
                                            mutation_by_replacement=True,
                                            num_genes=num_genes,
                                            init_range_low=init_range_low,
                                            init_range_high=init_range_high,
                                            random_mutation_min_val=random_mutation_min_val,
                                            random_mutation_max_val=random_mutation_max_val)

    assert num_duplicates == 0

def test_number_duplicates_int_gene_type_replacement_initial_population_multi_objective():
    num_genes = 10
    init_range_low = 0
    init_range_high = init_range_low + num_genes
    random_mutation_min_val = 0
    random_mutation_max_val = random_mutation_min_val + num_genes
    num_duplicates = number_duplicate_genes(gene_type=int,
                                            mutation_by_replacement=True,
                                            num_genes=num_genes,
                                            init_range_low=init_range_low,
                                            init_range_high=init_range_high,
                                            initial_population=initial_population,
                                            random_mutation_min_val=random_mutation_min_val,
                                            random_mutation_max_val=random_mutation_max_val)

    assert num_duplicates == 0

def test_number_duplicates_single_gene_space_multi_objective():
    num_duplicates = number_duplicate_genes(gene_space=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                            num_genes=10)

    assert num_duplicates == 0

def test_number_duplicates_single_gene_space_initial_population_multi_objective():
    num_duplicates = number_duplicate_genes(gene_space=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                            num_genes=10,
                                            initial_population=initial_population)

    assert num_duplicates == 0

def test_number_duplicates_single_range_gene_space_multi_objective():
    num_genes = 10
    num_duplicates = number_duplicate_genes(gene_space=range(num_genes),
                                            num_genes=num_genes)

    assert num_duplicates == 0

def test_number_duplicates_single_range_gene_space_initial_population_multi_objective():
    num_genes = 10
    num_duplicates = number_duplicate_genes(gene_space=range(num_genes),
                                            num_genes=num_genes,
                                            initial_population=initial_population)

    assert num_duplicates == 0

def test_number_duplicates_single_numpy_range_gene_space_multi_objective():
    num_genes = 10
    num_duplicates = number_duplicate_genes(gene_space=numpy.arange(num_genes),
                                            num_genes=num_genes)

    assert num_duplicates == 0

def test_number_duplicates_single_numpy_range_gene_space_initial_population_multi_objective():
    num_genes = 10
    num_duplicates = number_duplicate_genes(gene_space=numpy.arange(num_genes),
                                            num_genes=num_genes,
                                            initial_population=initial_population)

    assert num_duplicates == 0

def test_number_duplicates_nested_gene_space_multi_objective():
    num_duplicates = number_duplicate_genes(gene_space=[[0, 1], 
                                                        [1, 2], 
                                                        [2, 3],
                                                        [3, 4],
                                                        [4, 5],
                                                        [5, 6],
                                                        [6, 7],
                                                        [7, 8],
                                                        [8, 9],
                                                        [9, 10]],
                                            gene_type=int,
                                            num_genes=10)

    assert num_duplicates == 0

def test_number_duplicates_nested_gene_space_initial_population_multi_objective():
    num_duplicates = number_duplicate_genes(gene_space=[[0, 1], 
                                                        [1, 2], 
                                                        [2, 3],
                                                        [3, 4],
                                                        [4, 5],
                                                        [5, 6],
                                                        [6, 7],
                                                        [7, 8],
                                                        [8, 9],
                                                        [9, 10]],
                                            gene_type=int,
                                            num_genes=10,
                                            initial_population=initial_population)

    assert num_duplicates == 0


# def test_number_duplicates_nested_gene_space_nested_gene_type_multi_objective():
    """
    This example causes duplicate genes that can only be solved by changing the values of a chain of genes.
    Let's explain it using this solution: [0, 2, 3, 4, 5, 6, 6, 7, 8, 9]
    It has 2 genes with the value 6 at indices 5 and 6.
    According to the gene space, none of these genes can has a different value that solves the duplicates.
        -If the value of the gene at index 5 is changed from 6 to 5, then it causes another duplicate with the gene at index 4.
        -If the value of the gene at index 6 is changed from 6 to 7, then it causes another duplicate with the gene at index 7.
    The solution is to change a chain of genes that make a room to solve the duplicates between the 2 genes.
        1) Change the second gene from 2 to 1.
        2) Change the third gene from 3 to 2.
        3) Change the fourth gene from 4 to 3.
        4) Change the fifth gene from 5 to 4.
        5) Change the sixth gene from 6 to 5. This solves the duplicates.
    But this is NOT SUPPORTED yet.
    We support changing only a single gene that makes a room to solve the duplicates.

    Let's explain it using this solution: [1, 2, 2, 4, 5, 6, 6, 7, 8, 9]
    It has 2 genes with the value 2 at indices 1 and 2.
    This is how the duplicates are solved:
        1) Change the first gene from 1 to 0.
        2) Change the second gene from 2 to 1. This solves the duplicates.
    The result is [0, 1, 2, 4, 5, 6, 6, 7, 8, 9]
    """
    # num_duplicates = number_duplicate_genes(gene_space=[[0, 1], 
    #                                                     [1, 2], 
    #                                                     [2, 3],
    #                                                     [3, 4],
    #                                                     [4, 5],
    #                                                     [5, 6],
    #                                                     [6, 7],
    #                                                     [7, 8],
    #                                                     [8, 9],
    #                                                     [9, 10]],
    #                                         gene_type=[int, int, int, int, int, int, int, int, int, int],
    #                                         num_genes=10)

    # assert num_duplicates == 0

def test_number_duplicates_nested_gene_space_nested_gene_type_initial_population_multi_objective():
    num_duplicates = number_duplicate_genes(gene_space=[[0, 1], 
                                                        [1, 2], 
                                                        [2, 3],
                                                        [3, 4],
                                                        [4, 5],
                                                        [5, 6],
                                                        [6, 7],
                                                        [7, 8],
                                                        [8, 9],
                                                        [9, 10]],
                                            gene_type=[int, int, int, int, int, int, int, int, int, int],
                                            num_genes=10,
                                            initial_population=initial_population)

    assert num_duplicates == 0

if __name__ == "__main__":
    #### Single-objective
    print()
    test_number_duplicates_default()
    print()
    test_number_duplicates_default_initial_population()
    print()

    test_number_duplicates_float_gene_type()
    print()
    test_number_duplicates_float_gene_type_initial_population()
    print()

    test_number_duplicates_int_gene_type()
    print()
    test_number_duplicates_int_gene_type_initial_population()
    print()

    test_number_duplicates_int_gene_type_replacement()
    print()
    test_number_duplicates_int_gene_type_replacement_initial_population()
    print()

    test_number_duplicates_single_gene_space()
    print()
    test_number_duplicates_single_gene_space_initial_population()
    print()

    test_number_duplicates_single_range_gene_space()
    print()
    test_number_duplicates_single_range_gene_space_initial_population()
    print()

    test_number_duplicates_single_numpy_range_gene_space()
    print()
    test_number_duplicates_single_numpy_range_gene_space_initial_population()
    print()

    test_number_duplicates_nested_gene_space()
    print()
    test_number_duplicates_nested_gene_space_initial_population()
    print()

    # This example causes duplicates that can only be solved by changing a chain of genes.
    # test_number_duplicates_nested_gene_space_nested_gene_type()
    # print()
    test_number_duplicates_nested_gene_space_nested_gene_type_initial_population()
    print()

    #### Multi-objective
    print()
    test_number_duplicates_default_initial_population_multi_objective()
    print()

    test_number_duplicates_float_gene_type_multi_objective()
    print()
    test_number_duplicates_float_gene_type_initial_population_multi_objective()
    print()

    test_number_duplicates_int_gene_type_multi_objective()
    print()
    test_number_duplicates_int_gene_type_initial_population_multi_objective()
    print()

    test_number_duplicates_int_gene_type_replacement_multi_objective()
    print()
    test_number_duplicates_int_gene_type_replacement_initial_population_multi_objective()
    print()

    test_number_duplicates_single_gene_space_multi_objective()
    print()
    test_number_duplicates_single_gene_space_initial_population_multi_objective()
    print()

    test_number_duplicates_single_range_gene_space_multi_objective()
    print()
    test_number_duplicates_single_range_gene_space_initial_population_multi_objective()
    print()

    test_number_duplicates_single_numpy_range_gene_space_multi_objective()
    print()
    test_number_duplicates_single_numpy_range_gene_space_initial_population_multi_objective()
    print()

    test_number_duplicates_nested_gene_space_multi_objective()
    print()
    test_number_duplicates_nested_gene_space_initial_population_multi_objective()
    print()

    # This example causes duplicates that can only be solved by changing a chain of genes.
    # test_number_duplicates_nested_gene_space_nested_gene_type_multi_objective()
    # print()
    test_number_duplicates_nested_gene_space_nested_gene_type_initial_population_multi_objective()
    print()


