import pygad
import random
import numpy

num_generations = 100

def number_duplicate_genes(gene_space=None,
                           gene_type=float,
                           num_genes=10,
                           mutation_by_replacement=False,
                           random_mutation_min_val=-1,
                           random_mutation_max_val=1,
                           init_range_low=-4,
                           init_range_high=4):

    def fitness_func(ga, solution, idx):
        return random.random()

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=10,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           gene_type=gene_type,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           random_mutation_min_val=random_mutation_min_val,
                           random_mutation_max_val=random_mutation_max_val,
                           allow_duplicate_genes=False,
                           mutation_by_replacement=mutation_by_replacement,
                           save_solutions=True,
                           suppress_warnings=True)

    ga_instance.run()

    num_duplicates = 0
    for solution in ga_instance.solutions:
        num = len(solution) - len(set(solution))
        if num != 0:
            print(solution)
        num_duplicates += num

    print("Number of duplicates is {num_duplicates}.".format(num_duplicates=num_duplicates))
    return num_duplicates

def test_number_duplicates_default():
    num_duplicates = number_duplicate_genes()
    
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

def test_number_duplicates_single_gene_space():
    num_duplicates = number_duplicate_genes(gene_space=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                            num_genes=10)

    assert num_duplicates == 0

def test_number_duplicates_single_range_gene_space():
    num_genes = 10
    num_duplicates = number_duplicate_genes(gene_space=range(num_genes),
                                            num_genes=num_genes)

    assert num_duplicates == 0

def test_number_duplicates_single_numpy_range_gene_space():
    num_genes = 10
    num_duplicates = number_duplicate_genes(gene_space=numpy.arange(num_genes),
                                            num_genes=num_genes)

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
                                            random_mutation_min_val=11,
                                            random_mutation_max_val=20,
                                            mutation_by_replacement=True,
                                            num_genes=10)

    assert num_duplicates == 0

if __name__ == "__main__":
    # print()
    # test_number_duplicates_default()
    # print()
    # test_number_duplicates_float_gene_type()
    # print()
    # test_number_duplicates_int_gene_type()
    # print()
    # test_number_duplicates_single_gene_space()
    # print()
    # test_number_duplicates_single_range_gene_space()
    # print()
    # test_number_duplicates_single_numpy_range_gene_space()
    print()
    test_number_duplicates_nested_gene_space()
    print()
    
