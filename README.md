# GeneticAlgorithmPython
This project implements the genetic algorithm (GA) in Python mainly using NumPy.

The project has 2 main files which are:

1. `ga.py`: Holds all necessary methods for implementing the GA.

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
4. Train the genetic algorithm.

Let's discuss how to do each of these steps.

### Preparing the Parameters

Before running the GA, some parameters are required such as:

- `equation_inputs` : Inputs of the function to be optimized.
- `equation_output`: Function output.
- `sol_per_pop` : Number of solutions in the population.
  `num_parents_mating`  : Number of solutions to be selected as parents in the mating pool.
  `num_generations` : Number of generations.
- `mutation_percent_genes` : Percentage of genes to mutate.
- `mutation_num_genes` : Number of genes to mutate. If only the `mutation_percent_genes` argument is specified, then the value of `mutation_num_genes` will be implicitly calculated.

Here is the code for preparing such parameters:

```python
function_inputs = [4,-2,3.5,5,-11,-4.7]
function_output = 44

sol_per_pop = 8
num_parents_mating = 4
num_generations = 50

mutation_percent_genes=10
```

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
          function_inputs=function_inputs,
          function_output=function_output,
          mutation_percent_genes=10)
```

### Train the Genetic Algorithm

After an instance of the `GA` class is created, the next step is to call the `train()` method as follows:

```python
ga_instance.train()
```

Inside this method, the genetic algorithm evolves over a number of generations by doing the following tasks:

1. Calculating the fitness values of the solutions within the current population.
2. Select the best solutions as parents in the mating pool.
3. Apply the crossover & mutation operation
4. Repeat the process for the specified number of generations. 

### Plotting Results

There is a method named `plot_result()` which creates 2 figures summarizing the results.

```python
ga_instance.plot_result()
```

The first figure shows how the solutions' outputs change with the generations.

![Fig01](https://user-images.githubusercontent.com/16560492/78829951-8391d400-79e7-11ea-8edf-e46932dc76da.png)

The second figure shows how the fitness values of the solutions change with the generations.

![Fig02](https://user-images.githubusercontent.com/16560492/78830005-93111d00-79e7-11ea-9d8e-a8d8325a6101.png)

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

It is important to note that this project does not implement everything in GA and there are a wide number of variations to be applied. For example, this project uses decimal representation for the chromosome and the binary representations might be preferred for other problems.

## For Contacting the Author
* E-mail: ahmed.f.gad@gmail.com
* [LinkedIn](https://www.linkedin.com/in/ahmedfgad)
* [Amazon Author Page](https://amazon.com/author/ahmedgad)
* [Hearbeat](https://heartbeat.fritz.ai/@ahmedfgad)
* [Paperspace](https://blog.paperspace.com/author/ahmed)
* [KDnuggets](https://kdnuggets.com/author/ahmed-gad)
* [TowardsDataScience](https://towardsdatascience.com/@ahmedfgad)
* [GitHub](https://github.com/ahmedfgad)
