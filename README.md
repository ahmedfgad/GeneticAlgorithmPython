# PyGAD:  Genetic Algorithm in Python

[PyGAD](https://pypi.org/project/pygad) is an open-source easy-to-use Python 3 library for building the genetic algorithm and optimizing machine learning algorithms. It supports Keras and PyTorch. PyGAD supports optimizing both single-objective and multi-objective problems.

Check documentation of the [PyGAD](https://pygad.readthedocs.io/en/latest).

[![Downloads](https://pepy.tech/badge/pygad)](https://pepy.tech/project/pygad) [![PyPI version](https://badge.fury.io/py/pygad.svg)](https://badge.fury.io/py/pygad) ![Docs](https://readthedocs.org/projects/pygad/badge) [![PyGAD PyTest / Python 3.11](https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/workflows/main_py311.yml/badge.svg)](https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/workflows/main_py311.yml) [![PyGAD PyTest / Python 3.10](https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/workflows/main_py310.yml/badge.svg)](https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/workflows/main_py310.yml) [![PyGAD PyTest / Python 3.9](https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/workflows/main_py39.yml/badge.svg)](https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/workflows/main_py39.yml) [![PyGAD PyTest / Python 3.8](https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/workflows/main_py38.yml/badge.svg)](https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/workflows/main_py38.yml) [![PyGAD PyTest / Python 3.7](https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/workflows/main_py37.yml/badge.svg)](https://github.com/ahmedfgad/GeneticAlgorithmPython/actions/workflows/main_py37.yml) [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Translation](https://hosted.weblate.org/widgets/weblate/-/svg-badge.svg)](https://hosted.weblate.org/engage/weblate/) [![REUSE](https://api.reuse.software/badge/github.com/WeblateOrg/weblate)](https://api.reuse.software/info/github.com/WeblateOrg/weblate)

![PYGAD-LOGO](data:image/jpeg;base64,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)

[PyGAD](https://pypi.org/project/pygad) supports different types of crossover, mutation, and parent selection. [PyGAD](https://pypi.org/project/pygad) allows different types of problems to be optimized using the genetic algorithm by customizing the fitness function. 

The library is under active development and more features are added regularly. If you want a feature to be supported, please check the **Contact Us** section to send a request.

# Donation

* [Credit/Debit Card](https://donate.stripe.com/eVa5kO866elKgM0144): https://donate.stripe.com/eVa5kO866elKgM0144
* [Open Collective](https://opencollective.com/pygad): [opencollective.com/pygad](https://opencollective.com/pygad)
* PayPal: Use either this link: [paypal.me/ahmedfgad](https://paypal.me/ahmedfgad) or the e-mail address ahmed.f.gad@gmail.com
* Interac e-Transfer: Use e-mail address ahmed.f.gad@gmail.com

# Installation

To install [PyGAD](https://pypi.org/project/pygad), simply use pip to download and install the library from [PyPI](https://pypi.org/project/pygad) (Python Package Index). The library is at PyPI at this page https://pypi.org/project/pygad.

Install PyGAD with the following command:

```python
pip install pygad
```

To get started with PyGAD, please read the documentation at [Read The Docs](https://pygad.readthedocs.io/) https://pygad.readthedocs.io.

# PyGAD Source Code

The source code of the PyGAD' modules is found in the following GitHub projects:

- [pygad](https://github.com/ahmedfgad/GeneticAlgorithmPython): (https://github.com/ahmedfgad/GeneticAlgorithmPython)
- [pygad.nn](https://github.com/ahmedfgad/NumPyANN): https://github.com/ahmedfgad/NumPyANN
- [pygad.gann](https://github.com/ahmedfgad/NeuralGenetic): https://github.com/ahmedfgad/NeuralGenetic
- [pygad.cnn](https://github.com/ahmedfgad/NumPyCNN): https://github.com/ahmedfgad/NumPyCNN
- [pygad.gacnn](https://github.com/ahmedfgad/CNNGenetic): https://github.com/ahmedfgad/CNNGenetic
- [pygad.kerasga](https://github.com/ahmedfgad/KerasGA): https://github.com/ahmedfgad/KerasGA
- [pygad.torchga](https://github.com/ahmedfgad/TorchGA): https://github.com/ahmedfgad/TorchGA

The documentation of PyGAD is available at [Read The Docs](https://pygad.readthedocs.io/) https://pygad.readthedocs.io.

# PyGAD Documentation

The documentation of the PyGAD library is available at [Read The Docs](https://pygad.readthedocs.io) at this link: https://pygad.readthedocs.io. It discusses the modules supported by PyGAD, all its classes, methods, attribute, and functions. For each module, a number of examples are given.

If there is an issue using PyGAD, feel free to post at issue in this [GitHub repository](https://github.com/ahmedfgad/GeneticAlgorithmPython) https://github.com/ahmedfgad/GeneticAlgorithmPython or by sending an e-mail to ahmed.f.gad@gmail.com. 

If you built a project that uses PyGAD, then please drop an e-mail to ahmed.f.gad@gmail.com with the following information so that your project is included in the documentation.

- Project title
- Brief description
- Preferably, a link that directs the readers to your project

Please check the **Contact Us** section for more contact details.

# Life Cycle of PyGAD

The next figure lists the different stages in the lifecycle of an instance of the `pygad.GA` class. Note that PyGAD stops when either all generations are completed or when the function passed to the `on_generation` parameter returns the string `stop`.

![PyGAD Lifecycle](https://user-images.githubusercontent.com/16560492/220486073-c5b6089d-81e4-44d9-a53c-385f479a7273.jpg)

The next code implements all the callback functions to trace the execution of the genetic algorithm. Each callback function prints its name.

```python
import pygad
import numpy

function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

fitness_function = fitness_func

def on_start(ga_instance):
    print("on_start()")

def on_fitness(ga_instance, population_fitness):
    print("on_fitness()")

def on_parents(ga_instance, selected_parents):
    print("on_parents()")

def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()")

def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()")

def on_generation(ga_instance):
    print("on_generation()")

def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")

ga_instance = pygad.GA(num_generations=3,
                       num_parents_mating=5,
                       fitness_func=fitness_function,
                       sol_per_pop=10,
                       num_genes=len(function_inputs),
                       on_start=on_start,
                       on_fitness=on_fitness,
                       on_parents=on_parents,
                       on_crossover=on_crossover,
                       on_mutation=on_mutation,
                       on_generation=on_generation,
                       on_stop=on_stop)

ga_instance.run()
```

Based on the used 3 generations as assigned to the `num_generations` argument, here is the output.

```
on_start()

on_fitness()
on_parents()
on_crossover()
on_mutation()
on_generation()

on_fitness()
on_parents()
on_crossover()
on_mutation()
on_generation()

on_fitness()
on_parents()
on_crossover()
on_mutation()
on_generation()

on_stop()
```

# Example

Check the [PyGAD's documentation](https://pygad.readthedocs.io/en/latest/pygad.html) for information about the implementation of this example. It solves a single-objective problem.

```python
import pygad
import numpy

"""
Given the following function:
    y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44
What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize this function.
"""

function_inputs = [4,-2,3.5,5,-11,-4.7] # Function inputs.
desired_output = 44 # Function output.

def fitness_func(ga_instance, solution, solution_idx):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness

fitness_function = fitness_func

num_generations = 100 # Number of generations.
num_parents_mating = 7 # Number of solutions to be selected as parents in the mating pool.

# To prepare the initial population, there are 2 ways:
# 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
# 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
sol_per_pop = 50 # Number of solutions in the population.
num_genes = len(function_inputs)

last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")
    print(f"Change     = {ga_instance.best_solution()[1] - last_fitness}")
    last_fitness = ga_instance.best_solution()[1]

# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating, 
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop, 
                       num_genes=num_genes,
                       on_generation=callback_generation)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
ga_instance.plot_fitness()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

prediction = numpy.sum(numpy.array(function_inputs)*solution)
print(f"Predicted output based on the best solution : {prediction}")

if ga_instance.best_solution_generation != -1:
    print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

# Saving the GA instance.
filename = 'genetic' # The filename to which the instance is saved. The name is without extension.
ga_instance.save(filename=filename)

# Loading the saved GA instance.
loaded_ga_instance = pygad.load(filename=filename)
loaded_ga_instance.plot_fitness()
```

# For More Information

There are different resources that can be used to get started with the genetic algorithm and building it in Python. 

## Tutorial: Implementing Genetic Algorithm in Python

To start with coding the genetic algorithm, you can check the tutorial titled [**Genetic Algorithm Implementation in Python**](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad) available at these links:

- [LinkedIn](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad)
- [Towards Data Science](https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6)
- [KDnuggets](https://www.kdnuggets.com/2018/07/genetic-algorithm-implementation-python.html)

[This tutorial](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad) is prepared based on a previous version of the project but it still a good resource to start with coding the genetic algorithm.

[![Genetic Algorithm Implementation in Python](https://user-images.githubusercontent.com/16560492/78830052-a3c19300-79e7-11ea-8b9b-4b343ea4049c.png)](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad)

## Tutorial: Introduction to Genetic Algorithm

Get started with the genetic algorithm by reading the tutorial titled [**Introduction to Optimization with Genetic Algorithm**](https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad) which is available at these links:

* [LinkedIn](https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad)
* [Towards Data Science](https://towardsdatascience.com/introduction-to-optimization-with-genetic-algorithm-2f5001d9964b)
* [KDnuggets](https://www.kdnuggets.com/2018/03/introduction-optimization-with-genetic-algorithm.html)

[![Introduction to Genetic Algorithm](https://user-images.githubusercontent.com/16560492/82078259-26252d00-96e1-11ea-9a02-52a99e1054b9.jpg)](https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad)

## Tutorial: Build Neural Networks in Python

Read about building neural networks in Python through the tutorial titled [**Artificial Neural Network Implementation using NumPy and Classification of the Fruits360 Image Dataset**](https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad) available at these links:

* [LinkedIn](https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad)
* [Towards Data Science](https://towardsdatascience.com/artificial-neural-network-implementation-using-numpy-and-classification-of-the-fruits360-image-3c56affa4491)
* [KDnuggets](https://www.kdnuggets.com/2019/02/artificial-neural-network-implementation-using-numpy-and-image-classification.html)

[![Building Neural Networks Python](https://user-images.githubusercontent.com/16560492/82078281-30472b80-96e1-11ea-8017-6a1f4383d602.jpg)](https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad)

## Tutorial: Optimize Neural Networks with Genetic Algorithm

Read about training neural networks using the genetic algorithm through the tutorial titled [**Artificial Neural Networks Optimization using Genetic Algorithm with Python**](https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad) available at these links:

- [LinkedIn](https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad)
- [Towards Data Science](https://towardsdatascience.com/artificial-neural-networks-optimization-using-genetic-algorithm-with-python-1fe8ed17733e)
- [KDnuggets](https://www.kdnuggets.com/2019/03/artificial-neural-networks-optimization-genetic-algorithm-python.html)

[![Training Neural Networks using Genetic Algorithm Python](https://user-images.githubusercontent.com/16560492/82078300-376e3980-96e1-11ea-821c-aa6b8ceb44d4.jpg)](https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad)

## Tutorial: Building CNN in Python

To start with coding the genetic algorithm, you can check the tutorial titled [**Building Convolutional Neural Network using NumPy from Scratch**](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad) available at these links:

- [LinkedIn](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad)
- [Towards Data Science](https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a)
- [KDnuggets](https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html)
- [Chinese Translation](http://m.aliyun.com/yunqi/articles/585741)

[This tutorial](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad)) is prepared based on a previous version of the project but it still a good resource to start with coding CNNs.

[![Building CNN in Python](https://user-images.githubusercontent.com/16560492/82431022-6c3a1200-9a8e-11ea-8f1b-b055196d76e3.png)](https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad)

## Tutorial: Derivation of CNN from FCNN

Get started with the genetic algorithm by reading the tutorial titled [**Derivation of Convolutional Neural Network from Fully Connected Network Step-By-Step**](https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad) which is available at these links:

* [LinkedIn](https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad)
* [Towards Data Science](https://towardsdatascience.com/derivation-of-convolutional-neural-network-from-fully-connected-network-step-by-step-b42ebafa5275)
* [KDnuggets](https://www.kdnuggets.com/2018/04/derivation-convolutional-neural-network-fully-connected-step-by-step.html)

[![Derivation of CNN from FCNN](https://user-images.githubusercontent.com/16560492/82431369-db176b00-9a8e-11ea-99bd-e845192873fc.png)](https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad)

## Book: Practical Computer Vision Applications Using Deep Learning with CNNs

You can also check my book cited as [**Ahmed Fawzy Gad 'Practical Computer Vision Applications Using Deep Learning with CNNs'. Dec. 2018, Apress, 978-1-4842-4167-7**](https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665) which discusses neural networks, convolutional neural networks, deep learning, genetic algorithm, and more.

Find the book at these links:

- [Amazon](https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665)
- [Springer](https://link.springer.com/book/10.1007/978-1-4842-4167-7)
- [Apress](https://www.apress.com/gp/book/9781484241660)
- [O'Reilly](https://www.oreilly.com/library/view/practical-computer-vision/9781484241677)
- [Google Books](https://books.google.com.eg/books?id=xLd9DwAAQBAJ)

![Fig04](https://user-images.githubusercontent.com/16560492/78830077-ae7c2800-79e7-11ea-980b-53b6bd879eeb.jpg)

# Citing PyGAD - Bibtex Formatted Citation

If you used PyGAD, please consider adding a citation to the following paper about PyGAD:

```
@article{gad2023pygad,
  title={Pygad: An intuitive genetic algorithm python library},
  author={Gad, Ahmed Fawzy},
  journal={Multimedia Tools and Applications},
  pages={1--14},
  year={2023},
  publisher={Springer}
}
```

# Contact Us

* E-mail: ahmed.f.gad@gmail.com
* [LinkedIn](https://www.linkedin.com/in/ahmedfgad)
* [Paperspace](https://blog.paperspace.com/author/ahmed)
* [KDnuggets](https://kdnuggets.com/author/ahmed-gad)
* [TowardsDataScience](https://towardsdatascience.com/@ahmedfgad)
* [GitHub](https://github.com/ahmedfgad)
