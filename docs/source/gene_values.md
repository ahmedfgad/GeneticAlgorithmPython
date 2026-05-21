# Controlling Gene Values

This page covers the parameters that control the values a gene can take: the `gene_space` and `gene_type` parameters, gene constraints, the `sample_size` parameter, and preventing duplicate genes.

## Limit the Gene Value Range using the `gene_space` Parameter

In [PyGAD 2.11.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-11-0), the `gene_space` parameter added a new feature that lets you customize the range of accepted values for each gene. Let us first review the `gene_space` parameter and build on it.

The `gene_space` parameter lets you set the space of values for each gene. This way, the accepted values for each gene are restricted to the user-defined values. Assume there is a problem with 3 genes, where each gene has a different set of values:

1. Gene 1: `[0.4, 12, -5, 21.2]`
2. Gene 2: `[-2, 0.3]`
3. Gene 3: `[1.2, 63.2, 7.4]`

Then, the `gene_space` for this problem is as given below. Note that the order is very important.

```python
gene_space = [[0.4, 12, -5, 21.2],
              [-2, 0.3],
              [1.2, 63.2, 7.4]]
```

If all genes share the same set of values, then pass a single list to the `gene_space` parameter as follows. In this case, all genes can only take values from this list of 6 values.

```python
gene_space = [33, 7, 0.5, 95, 6.3, 0.74]
```

The previous example restricts the gene values to a fixed set of discrete values. If you want to use a range of discrete values for the gene, then you can use the `range()` function. For example, `range(1, 7)` means the allowed values for the gene are `1, 2, 3, 4, 5, and 6`. You can also use the `numpy.arange()` or `numpy.linspace()` functions for the same purpose.

The previous examples only work with discrete values, not continuous ones. In [PyGAD 2.11.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-11-0), the `gene_space` parameter can be assigned a dictionary that allows the gene to take values from a continuous range.

Assuming you want to restrict the gene within this half-open range [1 to 5) where 1 is included and 5 is not. Then simply create a dictionary with 2 items where the keys of the 2 items are:

1. `'low'`: The minimum value in the range which is 1 in the example.
2. `'high'`: The maximum value in the range which is 5 in the example.

The dictionary will look like that:

```python
{'low': 1,
 'high': 5}
```

It is not acceptable to add more than 2 items in the dictionary or use other keys than `'low'` and `'high'`.

For a 3-gene problem, the next code creates a dictionary for each gene to restrict its values in a continuous range. For the first gene, it can take any floating-point value from the range that starts from 1 (inclusive) and ends at 5 (exclusive).

```python
gene_space = [{'low': 1, 'high': 5}, {'low': 0.3, 'high': 1.4}, {'low': -0.2, 'high': 4.5}]
```

## More about the `gene_space` Parameter

The `gene_space` parameter customizes the space of values of each gene.  

Assuming that all genes have the same global space which include the values 0.3, 5.2, -4, and 8, then those values can be assigned to the `gene_space` parameter as a list, tuple, or range. Here is a list assigned to this parameter. By doing that, then the gene values are restricted to those assigned to the `gene_space` parameter.

```python
gene_space = [0.3, 5.2, -4, 8]
```

If some genes have different spaces, then `gene_space` should accept a nested list or tuple. In this case, the elements could be:

1. Number (of `int`, `float`, or `NumPy` data types): A single value to be assigned to the gene. This means this gene will have the same value across all generations.
2. `list`, `tuple`, `numpy.ndarray`, or any range like `range`, `numpy.arange()`, or `numpy.linspace`: It holds the space for each individual gene. But this space is usually discrete. That is there is a set of finite values to select from.
3. `dict`: To sample a value for a gene from a continuous range. The dictionary must have 2 mandatory keys which are `"low"` and `"high"` in addition to an optional key which is `"step"`. A random value is returned between the values assigned to the items with `"low"` and `"high"` keys. If the `"step"` exists, then this works as the previous options (i.e. discrete set of values). 
4. `None`: A gene with its space set to `None` is initialized randomly from the range specified by the 2 parameters `init_range_low` and `init_range_high`. For mutation, its value is mutated based on a random value from the range specified by the 2 parameters `random_mutation_min_val` and `random_mutation_max_val`. If all elements in the `gene_space` parameter are `None`, the parameter will not have any effect.

Assuming that a chromosome has 2 genes and each gene has a different value space. Then the `gene_space` could be assigned a nested list/tuple where each element determines the space of a gene. 

According to the next code, the space of the first gene is `[0.4, -5]` which has 2 values and the space for the second gene is `[0.5, -3.2, 8.8, -9]` which has 4 values.

```python
gene_space = [[0.4, -5], [0.5, -3.2, 8.2, -9]]
```

For a 2 gene chromosome, if the first gene space is restricted to the discrete values from 0 to 4 and the second gene is restricted to the values from 10 to 19, then it could be specified according to the next code.

```python
gene_space = [range(5), range(10, 20)]
```

The `gene_space` can also be assigned to a single range, as given below, where the values of all genes are sampled from the same range.

```python
gene_space = numpy.arange(15)
```

 The `gene_space` can be assigned a dictionary to sample a value from a continuous range.

```python
gene_space = {"low": 4, "high": 30}
```

 A step also can be assigned to the dictionary. This works as if a range is used.

```python
gene_space = {"low": 4, "high": 30, "step": 2.5}
```

> Setting a `dict` like `{"low": 0, "high": 10}` in the `gene_space` means that random values from the continuous range [0, 10) are sampled. Note that `0` is included but `10` is not included while sampling. Thus, the maximum value that could be returned is less than `10` like `9.9999`. But if the user decided to round the genes using, for example, `[float, 2]`, then this value will become 10. So, the user should be careful to the inputs.

If a `None` is assigned to only a single gene, then its value will be randomly generated initially using the `init_range_low` and `init_range_high` parameters in the `pygad.GA` class's constructor. During mutation, the value is sampled from the range defined by the 2 parameters `random_mutation_min_val` and `random_mutation_max_val`. This is an example where the second gene is given a `None` value.

```python
gene_space = [range(5), None, numpy.linspace(10, 20, 300)]
```

If the user did not assign the initial population to the `initial_population` parameter, the initial population is created randomly based on the `gene_space` parameter. Moreover, the mutation is applied based on this parameter.

### How Mutation Works with the `gene_space` Parameter?

Mutation changes based on whether the `gene_space` has a continuous range or discrete set of values.

If a gene has its **static/discrete space** defined in the `gene_space` parameter, then mutation works by replacing the gene value by a value randomly selected from the gene space. This happens for both `int` and `float` data types.

For example, the following `gene_space` has the static space `[1, 2, 3]` defined for the first gene. So, this gene can only have a value out of these 3 values.

```python
Gene space: [[1, 2, 3],
             None]
Solution: [1, 5]
```

For a solution like `[1, 5]`, then mutation happens for the first gene by simply replacing its current value by a randomly selected value (other than its current value if possible). So, the value 1 will be replaced by either 2 or 3.

For the second gene, its space is set to `None`. So, traditional mutation happens for this gene by:

1. Generating a random value from the range defined by the `random_mutation_min_val` and `random_mutation_max_val` parameters.
2. Adding this random value to the current gene's value.

If its current value is 5 and the random value is `-0.5`, then the new value is 4.5. If the gene type is integer, then the value will be rounded.

On the other hand, if a gene has a **continuous space** defined in the `gene_space` parameter, then mutation occurs by adding a random value to the current gene value.

For example, the following `gene_space` has the continuous space defined by the dictionary `{'low': 1, 'high': 5}`. This applies to all genes. So, mutation is applied to one or more selected genes by adding a random value to the current gene value.

```python
Gene space: {'low': 1, 'high': 5}
Solution: [1.5, 3.4]
```

Assuming `random_mutation_min_val=-1` and `random_mutation_max_val=1`, then a random value such as `0.3` can be added to the gene(s) participating in mutation. If only the first gene is mutated, then its new value changes from `1.5` to `1.5+0.3=1.8`. Note that PyGAD verifies that the new value is within the range. In the worst scenarios, the value will be set to either boundary of the continuous range. For example, if the gene value is 1.5 and the random value is -0.55, then the new value is 0.95, which is smaller than the lower boundary 1. So, the gene value will be set to 1.

If the dictionary has a step like the example below, then it is considered a discrete range and mutation occurs by randomly selecting a value from the set of values. In other words, no random value is added to the gene value.

```python
Gene space: {'low': 1, 'high': 5, 'step': 0.5}
```

## Gene Constraint

In [PyGAD 3.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-5-0), a new parameter called `gene_constraint` is added to the constructor of the `pygad.GA` class. An instance attribute of the same name is created for any instance of the `pygad.GA` class.

The `gene_constraint` parameter allows the users to define constraints to be enforced (as much as possible) when selecting a value for a gene. For example, this constraint is enforced when applying mutation to make sure the new gene value after mutation meets the gene constraint.

The default value of this parameter is `None` which means no genes have constraints. It can be assigned a list but the length of this list must be equal to the number of genes as specified by the `num_genes` parameter.

When assigned a list, the allowed values for each element are:

1. `None`: No constraint for the gene.
2. `callable`: A callable/function that accepts 2 parameters:
   1. The solution where the gene exists.
   2. A list or NumPy array of candidate values for the gene.

It is the user's responsibility to build such callables to filter the passed list of values and return a new list with the values that meet the gene constraint. If no value meets the constraint, return an empty list or NumPy array.

For example, if the gene must be smaller than 5, then use this callable:

```python
lambda solution,values: [val for val in values if val<5]
```

The first parameter is the solution where the target gene exists. It is passed just in case you would like to compare the gene value with other genes. The second parameter is the list of candidate values for the gene. The objective of the lambda function is to filter the values and return only the valid values that are less than 5.

A lambda function is used in this case but we can use a regular function:

```python
def constraint_func(solution,values): 
    return [val for val in values if val<5]
```

Assuming `num_genes` is 2, then here is a valid value for the `gene_constraint` parameter. 

```python
import pygad

def fitness_func(...):
    ...
    return fitness

ga_instance = pygad.GA(
    num_genes=2,
    sample_size=200,
    ...
    gene_constraint=
    [
        lambda solution,values: [val for val in values if val<5],
        lambda solution,values: [val for val in values if val>[solution[0]]
    ]
)
```

The first lambda function filters the values for the first gene by only considering the gene values that are less than 5. If the passed values is `[-5, 2, 6, 13, 3, 4, 0]`, then the returned filtered values will be `[-5, 2, 3, 4, 0]`.

The constraint for the second gene makes sure the selected value is larger than the value of the first gene. Assuming the values for the 2 parameters are:

1. `solution=[1, 4]`
2. `values=[17, 2, -1, 0.5, -2.1, 1.4]`

Then the value of the first gene in the passed solution is `1`. By filtering the passed values using the callable corresponding to the second gene, then the returned values will be `[17, 2, 1.4]` because these are the only values that are larger than the first gene value of `1`.

Sometimes it is normal for PyGAD to fail to find a gene value that satisfies the constraint. For example, if the possible gene values are only `[20,30,40]` and the gene constraint restricts the values to be greater than 50, then it is impossible to meet the constraint.

For some other cases, the constraint can be met but with some changes. For example, increasing the range from which a value is sampled. If the `gene_space` is used and assigned `range(10)`, then the gene constraint can be met by using `range(50)` so that we can find values greater than 50.

Even if the gene space is already assigned `range(1000)`, it might still not find values that meet the constraints. This is because PyGAD samples a number of values equal to the `sample_size` parameter which defaults to *100*. 

Out of the range of *1000* numbers, all the 100 values might not be satisfying the constraint. This issue could be solved by simply assigning a larger value for the `sample_size` parameter.

> PyGAD does not yet handle the **dependencies** among the genes in the `gene_constraint` parameter. 
>
> This is an example where gene 0 depends on gene 1. To efficiently enforce the constraints, the constraint for gene 1 must be enforced first (if not `None`) then the constraint for gene 0. 
>
> ```python
>     gene_constraint=
>     [
>         lambda solution,values: [val for val in values if val<solution[1]],
>         lambda solution,values: [val for val in values if val>10]
>     ]
> ```
>
> PyGAD applies constraints sequentially, starting from the first gene to the last. To ensure correct behavior when genes depend on each other, structure your GA problem so that if gene X depends on gene Y, then gene Y appears earlier in the chromosome (solution) than gene X. As a result, its gene constraint will be earlier in the list.

### Full Example

For a full example, please check the [`examples/example_gene_constraint.py` script](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/master/examples/example_gene_constraint.py).

## `sample_size` Parameter

In [PyGAD 3.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-5-0), a new parameter called `sample_size`. It is used in some situations where PyGAD seeks a single value for a gene out of a range. Two of the important use cases are:

1. Find a unique value for the gene. This is when the `allow_duplicate_genes` parameter is set to `False` to reject the duplicate gene values within the same solution.
2. Find a value that satisfies the `gene_constraint` parameter.

Given that we are sampling values from a continuous range as defined by the 2 attributes:

1. `random_mutation_min_val=0`
2. `random_mutation_max_val=100`

PyGAD samples a fixed number of values out of this continuous range. The number of values in the sample is defined by the `sample_size` parameter which defaults to `100`.

If the objective is to find a unique value or enforce the gene constraint, then the 100 values are filtered to keep only the values that keep the gene unique or meet the constraint. 

Sometimes 100 values is not enough and PyGAD sometimes fails to find a good value. In this case, it is highly recommended to increase the `sample_size` parameter. This is to create a larger sample to increase the chance of finding a value that meets our objectives.

## Prevent Duplicates in Gene Values

In [PyGAD 2.13.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-13-0), a new bool parameter called `allow_duplicate_genes` is supported to control whether duplicates are supported in the chromosome or not. In other words, whether 2 or more genes might have the same exact value. 

If `allow_duplicate_genes=True` (which is the default case), genes may have the same value. If `allow_duplicate_genes=False`, then no 2 genes will have the same value given that there are enough unique values for the genes.

The next code gives an example to use the `allow_duplicate_genes` parameter. A callback generation function is implemented to print the population after each generation. 

```python
import pygad

def fitness_func(ga_instance, solution, solution_idx):
    return 0

def on_generation(ga):
    print("Generation", ga.generations_completed)
    print(ga.population)

ga_instance = pygad.GA(num_generations=5,
                       sol_per_pop=5,
                       num_genes=4,
                       mutation_num_genes=3,
                       random_mutation_min_val=-5,
                       random_mutation_max_val=5,
                       num_parents_mating=2,
                       fitness_func=fitness_func,
                       gene_type=int,
                       on_generation=on_generation,
                       sample_size=200,
                       allow_duplicate_genes=False)
ga_instance.run()
```

Here are the population after the 5 generations. Note how there are no duplicate values.

```python
Generation 1
[[ 2 -2 -3  3]
 [ 0  1  2  3]
 [ 5 -3  6  3]
 [-3  1 -2  4]
 [-1  0 -2  3]]
Generation 2
[[-1  0 -2  3]
 [-3  1 -2  4]
 [ 0 -3 -2  6]
 [-3  0 -2  3]
 [ 1 -4  2  4]]
Generation 3
[[ 1 -4  2  4]
 [-3  0 -2  3]
 [ 4  0 -2  1]
 [-4  0 -2 -3]
 [-4  2  0  3]]
Generation 4
[[-4  2  0  3]
 [-4  0 -2 -3]
 [-2  5  4 -3]
 [-1  2 -4  4]
 [-4  2  0 -3]]
Generation 5
[[-4  2  0 -3]
 [-1  2 -4  4]
 [ 3  4 -4  0]
 [-1  0  2 -2]
 [-4  2 -1  1]]
```

The `allow_duplicate_genes` parameter can be used together with the `gene_space` parameter. Here is an example where each of the 4 genes has the same space of 4 values (1, 2, 3, and 4).

```python
import pygad

def fitness_func(ga_instance, solution, solution_idx):
    return 0

def on_generation(ga):
    print("Generation", ga.generations_completed)
    print(ga.population)

ga_instance = pygad.GA(num_generations=1,
                       sol_per_pop=5,
                       num_genes=4,
                       num_parents_mating=2,
                       fitness_func=fitness_func,
                       gene_type=int,
                       gene_space=[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                       on_generation=on_generation,
                       sample_size=200,
                       allow_duplicate_genes=False)
ga_instance.run()
```

Even though all the genes share the same space of values, no 2 genes have the same value, as shown in the next output.

```python
Generation 1
[[2 3 1 4]
 [2 3 1 4]
 [2 4 1 3]
 [2 3 1 4]
 [1 3 2 4]]
Generation 2
[[1 3 2 4]
 [2 3 1 4]
 [1 3 2 4]
 [2 3 4 1]
 [1 3 4 2]]
Generation 3
[[1 3 4 2]
 [2 3 4 1]
 [1 3 4 2]
 [3 1 4 2]
 [3 2 4 1]]
Generation 4
[[3 2 4 1]
 [3 1 4 2]
 [3 2 4 1]
 [1 2 4 3]
 [1 3 4 2]]
Generation 5
[[1 3 4 2]
 [1 2 4 3]
 [2 1 4 3]
 [1 2 4 3]
 [1 2 4 3]]
```

You should give enough values for the genes so that PyGAD can find an alternative when a gene value duplicates another gene.

If PyGAD fails to find a unique gene value while there is still room to find one, then set the `sample_size` parameter to a larger value. Check the [sample_size Parameter](https://pygad.readthedocs.io/en/latest/gene_values.html#sample-size-parameter) section for more information.

### Limitation

There might be 2 duplicate genes where changing either of the 2 duplicating genes will not solve the problem. For example, if `gene_space=[[3, 0, 1], [4, 1, 2], [0, 2], [3, 2, 0]]` and the solution is `[3 2 0 0]`, then the values of the last 2 genes duplicate. There are no possible changes in the last 2 genes to solve the problem. 

This problem can be solved by randomly changing one of the non-duplicating genes to make room for a unique value in one of the 2 duplicating genes. For example, by changing the second gene from 2 to 4, then any of the last 2 genes can take the value 2 and solve the duplicates. The resultant gene is then `[3 4 2 0]`. But this option is not yet supported in PyGAD.

### Solve Duplicates using a Third Gene

When `allow_duplicate_genes=False` and a user-defined `gene_space` is used, it sometimes happens that there is no room to solve the duplicates between the 2 genes by simply replacing the value of one gene with another. In [PyGAD 3.1.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-1-0), the duplicates are solved by looking for a third gene that helps solve them. The following examples explain how it works.

Example 1:

Let's assume that this gene space is used and there is a solution with 2 duplicate genes with the same value 4.

```python
Gene space: [[2, 3],
             [3, 4],
             [4, 5],
             [5, 6]]
Solution: [3, 4, 4, 5]
```

By checking the gene space, the second gene can have the values `[3, 4]` and the third gene can have the values `[4, 5]`. To solve the duplicates, we change the value of one of these 2 genes.

If the value of the second gene changes from 4 to 3, then it will duplicate the first gene. If we change the value of the third gene from 4 to 5, then it will duplicate the fourth gene. In short, simply selecting a different value for either the second or third gene will introduce new duplicate genes.

When there are 2 duplicate genes but there is no way to solve their duplicates, then the solution is to change a third gene that makes a room to solve the duplicates between the 2 genes.

In our example, duplicates between the second and third genes can be solved by, for example,:

* Changing the first gene from 3 to 2 then changing the second gene from 4 to 3. 
* Or changing the fourth gene from 5 to 6 then changing the third gene from 4 to 5. 

Generally, this is how to solve such duplicates:

1. For any duplicate gene **GENE1**, select another value.
2. Check which other gene **GENEX** has duplicate with this new value.
3. Find if **GENEX** can have another value that will not cause any more duplicates. If so, go to step 7.
4. If all the other values of **GENEX** will cause duplicates, then try another gene **GENEY**.
5. Repeat steps 3 and 4 until exploring all the genes. 
6. If there is no way to solve the duplicates, then we have to keep the duplicate value.
7. If a value for a gene **GENEM** is found that will not cause more duplicates, then use this value for the gene **GENEM**.
8. Replace the value of the gene **GENE1** by the old value of the gene **GENEM**. This solves the duplicates.

This is an example to solve the duplicate for the solution `[3, 4, 4, 5]`:

1. Let's use the second gene with value 4. Because the space of this gene is `[3, 4]`, then the only other value we can select is 3.
2. The first gene also has the value 3.
3. The first gene has another value 2 that will not cause more duplicates in the solution. Then go to step 7.
4. Skip.
5. Skip.
6. Skip.
7. The value of the first gene 3 will be replaced by the new value 2. The new solution is [2, 4, 4, 5].
8. Replace the value of the second gene 4 by the old value of the first gene which is 3. The new solution is [2, 3, 4, 5]. The duplicate is solved.

Example 2:

```python
Gene space: [[0, 1], 
             [1, 2], 
             [2, 3],
             [3, 4]]
Solution: [1, 2, 2, 3]
```

The quick summary is:

* Change the value of the first gene from 1 to 0. The solution becomes [0, 2, 2, 3].
* Change the value of the second gene from 2 to 1. The solution becomes [0, 1, 2, 3]. The duplicate is solved.

## More about the `gene_type` Parameter

The `gene_type` parameter allows the user to control the data type for all genes at once or each individual gene. In [PyGAD 2.15.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0), the `gene_type` parameter also supports customizing the precision for `float` data types. As a result, the `gene_type` parameter helps to:

1. Select a data type for all genes with or without precision. 
2. Select a data type for each individual gene with or without precision. 

Let us look at some examples.

### Data Type for All Genes without Precision

The data type for all genes can be specified by assigning the numeric data type directly to the `gene_type` parameter. This is an example to make all genes of `int` data types.

```python
gene_type=int
```

Given that the supported numeric data types of PyGAD include Python's `int` and `float` in addition to all numeric types of `NumPy`, then any of these types can be assigned to the `gene_type` parameter.

If no precision is specified for a `float` data type, then the complete floating-point number is kept.

The next code uses an `int` data type for all genes where the genes in the initial and final population are only integers.

```python
import pygad
import numpy

equation_inputs = [4, -2, 3.5, 8, -2]
desired_output = 2671.1234

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution * equation_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

ga_instance = pygad.GA(num_generations=10,
                       sol_per_pop=5,
                       num_parents_mating=2,
                       num_genes=len(equation_inputs),
                       fitness_func=fitness_func,
                       gene_type=int)

print("Initial Population")
print(ga_instance.initial_population)

ga_instance.run()

print("Final Population")
print(ga_instance.population)
```

```python
Initial Population
[[ 1 -1  2  0 -3]
 [ 0 -2  0 -3 -1]
 [ 0 -1 -1  2  0]
 [-2  3 -2  3  3]
 [ 0  0  2 -2 -2]]

Final Population
[[ 1 -1  2  2  0]
 [ 1 -1  2  2  0]
 [ 1 -1  2  2  0]
 [ 1 -1  2  2  0]
 [ 1 -1  2  2  0]]
```

### Data Type for All Genes with Precision

A precision can only be specified for a `float` data type and cannot be specified for integers. Here is an example to use a precision of 3 for the `float` data type. In this case, all genes are of type `float` and their maximum precision is 3. 

```python
gene_type=[float, 3]
```

The next code prints the initial and final population where the genes are of type `float` with precision 3.

```python
import pygad
import numpy

equation_inputs = [4, -2, 3.5, 8, -2]
desired_output = 2671.1234

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution * equation_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)

    return fitness

ga_instance = pygad.GA(num_generations=10,
                       sol_per_pop=5,
                       num_parents_mating=2,
                       num_genes=len(equation_inputs),
                       fitness_func=fitness_func,
                       gene_type=[float, 3])

print("Initial Population")
print(ga_instance.initial_population)

ga_instance.run()

print("Final Population")
print(ga_instance.population)
```

```python
Initial Population
[[-2.417 -0.487  3.623  2.457 -2.362]
 [-1.231  0.079 -1.63   1.629 -2.637]
 [ 0.692 -2.098  0.705  0.914 -3.633]
 [ 2.637 -1.339 -1.107 -0.781 -3.896]
 [-1.495  1.378 -1.026  3.522  2.379]]

Final Population
[[ 1.714 -1.024  3.623  3.185 -2.362]
 [ 0.692 -1.024  3.623  3.185 -2.362]
 [ 0.692 -1.024  3.623  3.375 -2.362]
 [ 0.692 -1.024  4.041  3.185 -2.362]
 [ 1.714 -0.644  3.623  3.185 -2.362]]
```

### Data Type for each Individual Gene without Precision

In [PyGAD 2.14.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-14-0), the `gene_type` parameter allows customizing the gene type for each individual gene. This is by using a `list`/`tuple`/`numpy.ndarray` with number of elements equal to the number of genes. For each element, a type is specified for the corresponding gene.

This is an example for a 5-gene problem where different types are assigned to the genes.

```python
gene_type=[int, float, numpy.float16, numpy.int8, float]
```

This is a complete code that prints the initial and final population for a custom-gene data type.

```python
import pygad
import numpy

equation_inputs = [4, -2, 3.5, 8, -2]
desired_output = 2671.1234

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution * equation_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

ga_instance = pygad.GA(num_generations=10,
                       sol_per_pop=5,
                       num_parents_mating=2,
                       num_genes=len(equation_inputs),
                       fitness_func=fitness_func,
                       gene_type=[int, float, numpy.float16, numpy.int8, float])

print("Initial Population")
print(ga_instance.initial_population)

ga_instance.run()

print("Final Population")
print(ga_instance.population)
```

```python
Initial Population
[[0 0.8615522360026828 0.7021484375 -2 3.5301821368185866]
 [-3 2.648189378595294 -3.830078125 1 -0.9586271572917742]
 [3 3.7729827570110714 1.2529296875 -3 1.395741994211889]
 [0 1.0490687178053282 1.51953125 -2 0.7243617940450235]
 [0 -0.6550158436937226 -2.861328125 -2 1.8212734549263097]]

Final Population
[[3 3.7729827570110714 2.055 0 0.7243617940450235]
 [3 3.7729827570110714 1.458 0 -0.14638754050305036]
 [3 3.7729827570110714 1.458 0 0.0869406120516778]
 [3 3.7729827570110714 1.458 0 0.7243617940450235]
 [3 3.7729827570110714 1.458 0 -0.14638754050305036]]
```

### Data Type for each Individual Gene with Precision

The precision can also be specified for the `float` data types as in the next line where the second gene precision is 2 and last gene precision is 1.

```python
gene_type=[int, [float, 2], numpy.float16, numpy.int8, [float, 1]]
```

This is a complete example where the initial and final populations are printed where the genes comply with the data types and precisions specified.

```python
import pygad
import numpy

equation_inputs = [4, -2, 3.5, 8, -2]
desired_output = 2671.1234

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution * equation_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

ga_instance = pygad.GA(num_generations=10,
                       sol_per_pop=5,
                       num_parents_mating=2,
                       num_genes=len(equation_inputs),
                       fitness_func=fitness_func,
                       gene_type=[int, [float, 2], numpy.float16, numpy.int8, [float, 1]])

print("Initial Population")
print(ga_instance.initial_population)

ga_instance.run()

print("Final Population")
print(ga_instance.population)
```

```python
Initial Population
[[-2 -1.22 1.716796875 -1 0.2]
 [-1 -1.58 -3.091796875 0 -1.3]
 [3 3.35 -0.107421875 1 -3.3]
 [-2 -3.58 -1.779296875 0 0.6]
 [2 -3.73 2.65234375 3 -0.5]]

Final Population
[[2 -4.22 3.47 3 -1.3]
 [2 -3.73 3.47 3 -1.3]
 [2 -4.22 3.47 2 -1.3]
 [2 -4.58 3.47 3 -1.3]
 [2 -3.73 3.47 3 -1.3]]
```
