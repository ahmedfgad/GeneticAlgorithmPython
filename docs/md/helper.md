# `pygad.helper` Module

This section of the PyGAD's library documentation discusses the `pygad.helper` module. 

The `pygad.helper` module has 2 submodules:

1. `pygad.helper.unique`: A module of methods for creating unique genes.
2. `pygad.helper.misc`: A module of miscellaneous helper methods.

## `pygad.helper.unique` Module

The `pygad.helper.unique` module has a class named `Unique` with the following helper methods. Such methods help to check and fix duplicate values in the genes of a solution.

1. `solve_duplicate_genes_randomly()`: Solves the duplicates in a solution by randomly selecting new values for the duplicating genes.
2. `solve_duplicate_genes_by_space()`: Solves the duplicates in a solution by selecting values for the duplicating genes from the gene space
3. `unique_int_gene_from_range()`: Finds a unique integer value for the gene out of a range defined by start and end points.
4. `unique_float_gene_from_range()`: Finds a unique float value for the gene out of a range defined by start and end points.
5. `select_unique_value()`: Selects a unique value (if possible) from a list of gene values.
6. `unique_genes_by_space()`: Loops through all the duplicating genes to find unique values that from their gene spaces to solve the duplicates. For each duplicating gene, a call to the `unique_gene_by_space()` is made.
7. `unique_gene_by_space()`: Returns a unique gene value for a single gene based on its value space to solve the duplicates.
8. `find_two_duplicates()`: Identifies the first occurrence of a duplicate gene in the solution.
9. `unpack_gene_space()`: Unpacks the gene space for selecting a value to resolve duplicates by converting ranges into lists of values.
10. `solve_duplicates_deeply()`: Sometimes it is impossible to solve the duplicate genes by simply randomly selecting another value for either genes. This function solve the duplicates between 2 genes by searching for a third gene that can make assist in the solution.

## `pygad.helper.misc` Module

The `pygad.helper.misc` module has a class called `Helper` with some methods to help in different stages of the GA pipeline. It is introduced in [PyGAD 3.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-5-0).

1. `change_population_dtype_and_round()`: For each gene in the population, round the gene value and change the data type.
2. `change_gene_dtype_and_round()`: Round the change the data type of a single gene.
3. `mutation_change_gene_dtype_and_round()`: Decides whether mutation is done by replacement or not. Then it rounds and change the data type of the new gene value.
4. `validate_gene_constraint_callable_output()`: Validates the output of the user-defined callable/function that checks whether the gene constraint defined in the `gene_constraint` parameter is satisfied or not.
5. `get_gene_dtype()`: Returns the gene data type from the `gene_type` instance attribute.
6. `get_random_mutation_range()`: Returns the random mutation range using the `random_mutation_min_val` and `random_mutation_min_val` instance attributes.
7. `get_initial_population_range()`: Returns the initial population values range using the `init_range_low` and `init_range_high` instance attributes.
8. `generate_gene_value_from_space()`: Generates/selects a value for a gene using the `gene_space` instance attribute.
9. `generate_gene_value_randomly()`: Generates a random value for the gene. Only used if `gene_space` is `None`.
10. `generate_gene_value()`: Generates a value for the gene. It checks whether `gene_space` is `None` and calls either `generate_gene_value_randomly()` or `generate_gene_value_from_space()`.
11. `filter_gene_values_by_constraint()`: Receives a list of values for a gene. Then it filters such values using the gene constraint.
12. `get_valid_gene_constraint_values()`: Selects one valid gene value that satisfy the gene constraint. It simply calls `generate_gene_value()` to generate some gene values then it filters such values using `filter_gene_values_by_constraint()`.

