# GeneticAlgorithmPython
This project implements the genetic algorithm (GA) in Python mainly using NumPy.

The project has 2 main files which are:

1. `ga.py`: Holds all necessary methods for implementing the genetic algorithm inside a class named `GA`.

2. `example.py`: Just gives an example of how to use the project by calling the methods in the `ga.py` file.

To test the project, you can simply run the `example.py` file.

```
python example.py
```

## How to Use the Project?

To use the project, here is the summary of the minimum required steps: 

1. Prepare the required parameters.
2. Import the `ga.py` module.
3. Create an instance of the `GA` class.
4. Run the genetic algorithm.
5. Plotting Results.
6. Saving & Loading the Results.

Let's discuss how to do each of these steps.

### The Supported Parameters

The project has many parameters to allow customizing the genetic algorithm for your purpose. Before running the GA, the parameters must be prepared. The list of all supported parameters is as follows:

- `num_generations` : Number of generations.
- `sol_per_pop` : Number of solutions (i.e. chromosomes) within the population.
- `num_parents_mating ` : Number of solutions to be selected as parents.
- `num_genes`: Number of genes in the solution/chromosome.
- `fitness_func` : A function for calculating the fitness value for each solution.
- `parent_selection_type="sss"` : The parent selection type. Supported types are `sss` (for steady state selection), `rws` (for roulette wheel selection), `sus` (for stochastic universal selection), `rank` (for rank selection), `random` (for random selection), and `tournament` (for tournament selection).
- `keep_parents=-1` : Number of parents to keep in the current population. `-1` (default) means keep all parents in the next population. `0` means keep no parents in the next population. A value `greater than 0` means keep the specified number of parents in the next population. Note that the value assigned to `keep_parents` cannot be `< - 1` or greater than the number of solutions within the population `sol_per_pop`.
- `K_tournament=3` : In case that the parent selection type is `tournament`, the `K_tournament` specifies the number of parents participating in the tournament selection. It defaults to `3`.
- `crossover_type="single_point"` : Type of the crossover operation. Supported types are `single_point` (for single point crossover), `two_points` (for two points crossover), and `uniform` (for uniform crossover). It defaults to `single_point`.
- `mutation_type="random"` : Type of the mutation operation. Supported types are `random` (for random mutation), `swap` (for swap mutation), `inversion` (for inversion mutation), and `scramble` (for scramble mutation). It defaults to `random`.
- `mutation_percent_genes=10` : Percentage of genes to mutate which defaults to `10`. Out of this percentage, the number of genes to mutate is deduced. This parameter has no action if the parameter `mutation_num_genes` exists. 
- `mutation_num_genes=None` : Number of genes to mutate which defaults to `None` meaning that no number is specified. If the parameter `mutation_num_genes` exists, then no need for the parameter `mutation_percent_genes`.
- `random_mutation_min_val=-1.0` : For `random` mutation, the `random_mutation_min_val` parameter specifies the start value of the range from which a random value is selected to be added to the gene. It defaults to `-1`.
- `random_mutation_max_val=1.0` : For `random` mutation, the `random_mutation_max_val` parameter specifies the end value of the range from which a random value is selected to be added to the gene. It defaults to `+1`.

The user doesn't have to specify all of such parameters while creating an instance of the GA class. A very important parameter you must care about is `fitness_func`.

### Preparing the `fitness_func` Parameter

Even there are a number of steps in the genetic algorithm pipeline that can work the same regardless of the problem being solved, one critical step is the calculation of the fitness value. There is no unique way of calculating the fitness value and it changes from one problem to another. 

On **`15 April 2020`**, a new argument named `fitness_func` is added that allows the user to specify a custom function to be used as a fitness function. This function must be a **maximization function** so that a solution with a high fitness value returned is selected compared to a solution with a low value. Doing that allows the user to freely use the project to solve any problem by passing the appropriate fitness function. 

Let's discuss an example:

> Given the following function:
>     y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
>     where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44
> What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize this function.

So, the task is about using the genetic algorithm to find the best values for the 6 weight `W1` to `W6`. Thinking of the problem, it is clear that the best solution is that returning an output that is close to the desired output `y=44`. So, the fitness function should return a value that gets higher when the solution's output is closer to `y=44`. Here is a function that does that. The function must accept a single parameter which is a 1D vector representing a single solution.

```python
function_inputs = [4,-2,3.5,5,-11,-4.7] # Function inputs.
desired_output = 44 # Function output.

def fitness_func(solution):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness
```

By creating this function, you are ready to use the project. 

### Parameters Example

Here is an example for preparing the parameters:

```python
num_generations = 50
sol_per_pop = 8
num_parents_mating = 4

mutation_percent_genes = 10

parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"

keep_parents = 1

num_genes = len(function_inputs)
```

After the parameters are prepared, we can import the `ga` module and build an instance of the GA class.

### Import the `ga.py` Module

The next step is to import the `ga` module as follows:

```python
import ga
```

This module has a class named `GA` which holds the implementation of all methods for running the genetic algorithm.

### Create an Instance of the `GA` Class.

The `GA` class is instantiated where the previously prepared parameters are fed to its constructor. The constructor is responsible for creating the initial population.

```python
ga_instance = ga.GA(num_generations=num_generations, 
          sol_per_pop=sol_per_pop, 
          num_parents_mating=num_parents_mating, 
          num_genes=num_genes,
          fitness_func=fitness_func,
          mutation_percent_genes=mutation_percent_genes,
          mutation_num_genes=mutation_num_genes,
          parent_selection_type=parent_selection_type,
          crossover_type=crossover_type,
          mutation_type=mutation_type,
          keep_parents=keep_parents,
          K_tournament=3)
```

### Run the Genetic Algorithm

After an instance of the `GA` class is created, the next step is to call the `run()` method as follows:

```python
ga_instance.run()
```

Inside this method, the genetic algorithm evolves over a number of generations by doing the following tasks:

1. Calculating the fitness values of the solutions within the current population.
2. Select the best solutions as parents in the mating pool.
3. Apply the crossover & mutation operation
4. Repeat the process for the specified number of generations. 

### Plotting Results

There is a method named `plot_result()` which creates a figure summarizing how the fitness values of the solutions change with the generations .

```python
ga_instance.plot_result()
```

![Fig02](https://user-images.githubusercontent.com/16560492/78830005-93111d00-79e7-11ea-9d8e-a8d8325a6101.png)

### Saving & Loading the Results

After the `run()` method completes, it is possible to save the current instance of the genetic algorithm to avoid losing the progress made. The `save()` method is available for that purpose. According to the next code, a file named `genetic.pkl` will be created and saved in the current directory.

```python
# Saving the GA instance.
filename = 'genetic' # The filename to which the instance is saved. The name is without extension.
ga_instance.save(filename=filename)
```

You can also load the saved model using the `load()` function and continue using it. For example, you might run the genetic algorithm for a number of generations, save its current state using the `save()` method, load the model using the `load()` function, and then call the `run()` method again.

```python
# Loading the saved GA instance.
loaded_ga_instance = ga.load(filename=filename)
```

After the instance is loaded, you can use it to run any method or access any property.

```python
print(loaded_ga_instance.best_solution())
```

## Crossover, Mutation, and Parent Selection

The project supports different types for selecting the parents and applying the crossover & mutation operators.

The supported crossover operations at this time are:

- Single point.
- Two points.
- Uniform.

The supported mutation operations at this time are:

- Random
- Swap
- Inversion
- Scramble

The supported parent selection techniques at this time are:

- Steady state
- Roulette wheel
- Stochastic universal
- Rank
- Random
- Tournament

More types will be added in the future. You can also ask for supporting more types by opening an issue in the project.

## For More Information

To start with coding the genetic algorithm, you can check the tutorial titled [**Genetic Algorithm Implementation in Python**](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad) available at these links:

- https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad

- https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6

[This tutorial](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad) is prepared based on a previous version of the project but it still a good resource to start with coding the genetic algorithm.

![Fig03](https://user-images.githubusercontent.com/16560492/78830052-a3c19300-79e7-11ea-8b9b-4b343ea4049c.png)

You can also check my book cited as [**Ahmed Fawzy Gad 'Practical Computer Vision Applications Using Deep Learning with CNNs'. Dec. 2018, Apress, 978-1-4842-4167-7**](https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665).

![Fig04](https://user-images.githubusercontent.com/16560492/78830077-ae7c2800-79e7-11ea-980b-53b6bd879eeb.jpg)

---

**Important Note**

It is important to note that this project does not implement everything in GA and there are a wide number of variations to be applied. For example, this project just uses decimal representation for the chromosome and the binary representations might be preferred for other problems.

## Get it Touch
* E-mail: ahmed.f.gad@gmail.com
* [LinkedIn](https://www.linkedin.com/in/ahmedfgad)
* [Amazon Author Page](https://amazon.com/author/ahmedgad)
* [Heartbeat](https://heartbeat.fritz.ai/@ahmedfgad)
* [Paperspace](https://blog.paperspace.com/author/ahmed)
* [KDnuggets](https://kdnuggets.com/author/ahmed-gad)
* [TowardsDataScience](https://towardsdatascience.com/@ahmedfgad)
* [GitHub](https://github.com/ahmedfgad)
