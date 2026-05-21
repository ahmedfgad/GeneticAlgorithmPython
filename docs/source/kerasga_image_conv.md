# Example 4: Image Multi-Class Classification (Conv Layers)

Compared to the previous example that uses only dense layers, this example uses convolutional layers to classify the same dataset.

Here is the complete code.

```python
import tensorflow.keras
import pygad.kerasga
import numpy
import pygad

def fitness_func(ga_instance, solution, sol_idx):
    global data_inputs, data_outputs, keras_ga, model

    predictions = pygad.kerasga.predict(model=model,
                                        solution=solution,
                                        data=data_inputs)

    cce = tensorflow.keras.losses.CategoricalCrossentropy()
    solution_fitness = 1.0 / (cce(data_outputs, predictions).numpy() + 0.00000001)

    return solution_fitness

def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")

# Build the keras model using the functional API.
input_layer = tensorflow.keras.layers.Input(shape=(100, 100, 3))
conv_layer1 = tensorflow.keras.layers.Conv2D(filters=5,
                                             kernel_size=7,
                                             activation="relu")(input_layer)
max_pool1 = tensorflow.keras.layers.MaxPooling2D(pool_size=(5,5),
                                                 strides=5)(conv_layer1)
conv_layer2 = tensorflow.keras.layers.Conv2D(filters=3,
                                             kernel_size=3,
                                             activation="relu")(max_pool1)
flatten_layer  = tensorflow.keras.layers.Flatten()(conv_layer2)
dense_layer = tensorflow.keras.layers.Dense(15, activation="relu")(flatten_layer)
output_layer = tensorflow.keras.layers.Dense(4, activation="softmax")(dense_layer)

model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)

# Create an instance of the pygad.kerasga.KerasGA class to build the initial population.
keras_ga = pygad.kerasga.KerasGA(model=model,
                                 num_solutions=10)

# Data inputs
data_inputs = numpy.load("../data/dataset_inputs.npy")

# Data outputs
data_outputs = numpy.load("../data/dataset_outputs.npy")
data_outputs = tensorflow.keras.utils.to_categorical(data_outputs)

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/pygad.html#pygad-ga-class
num_generations = 200 # Number of generations.
num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
initial_population = keras_ga.population_weights # Initial population of network weights.

# Create an instance of the pygad.GA class
ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=on_generation)

# Start the genetic algorithm evolution.
ga_instance.run()

# After the generations complete, a plot is shown that summarizes how the fitness values evolve over the generations.
ga_instance.plot_fitness(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

# Make predictions based on the best solution.
predictions = pygad.kerasga.predict(model=model,
                                    solution=solution,
                                    data=data_inputs)
# print(f"Predictions : \n{predictions}")

# Calculate the categorical crossentropy for the trained model.
cce = tensorflow.keras.losses.CategoricalCrossentropy()
print(f"Categorical Crossentropy : {cce(data_outputs, predictions).numpy()}")

# Calculate the classification accuracy for the trained model.
ca = tensorflow.keras.metrics.CategoricalAccuracy()
ca.update_state(data_outputs, predictions)
accuracy = ca.result().numpy()
print(f"Accuracy : {accuracy}")
```

Compared to the previous example, the only change is that the architecture uses convolutional and max-pooling layers. The shape of each input sample is 100x100x3.

```python
# Build the keras model using the functional API.
input_layer = tensorflow.keras.layers.Input(shape=(100, 100, 3))
conv_layer1 = tensorflow.keras.layers.Conv2D(filters=5,
                                             kernel_size=7,
                                             activation="relu")(input_layer)
max_pool1 = tensorflow.keras.layers.MaxPooling2D(pool_size=(5,5),
                                                 strides=5)(conv_layer1)
conv_layer2 = tensorflow.keras.layers.Conv2D(filters=3,
                                             kernel_size=3,
                                             activation="relu")(max_pool1)
flatten_layer  = tensorflow.keras.layers.Flatten()(conv_layer2)
dense_layer = tensorflow.keras.layers.Dense(15, activation="relu")(flatten_layer)
output_layer = tensorflow.keras.layers.Dense(4, activation="softmax")(dense_layer)

model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)
```

## Prepare the Training Data

The data used in this example is available as 2 files:

1. [dataset_inputs.npy](https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_inputs.npy): Data inputs. https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_inputs.npy
2. [dataset_outputs.npy](https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_outputs.npy): Class labels. https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_outputs.npy

The data consists of 4 classes of images. The image shape is `(100, 100, 3)` and there are 20 images per class for a total of 80 training samples. For more information about the dataset, check the [Reading the Data](https://pygad.readthedocs.io/en/latest/cnn.html#reading-the-data) section of the `pygad.cnn` module.

Simply download these 2 files and read them according to the next code. Note that the class labels are one-hot encoded using the `tensorflow.keras.utils.to_categorical()` function.

```python
import numpy

data_inputs = numpy.load("../data/dataset_inputs.npy")

data_outputs = numpy.load("../data/dataset_outputs.npy")
data_outputs = tensorflow.keras.utils.to_categorical(data_outputs)
```

The next figure shows how the fitness value changes.

![pygad_keras_image_classification_Conv](https://user-images.githubusercontent.com/16560492/93722654-cc55d780-fb98-11ea-8f95-7b65dc67f5c8.png)

Here are some statistics about the trained model. The model accuracy is 75% after the 200 generations. Note that just running the code again may give different results.

```
Fitness value of the best solution = 2.7462310258668805
Index of the best solution : 0
Categorical Crossentropy :  0.3641354
Accuracy :  0.75
```

To improve the model performance, you can do the following:

- Add more layers 
- Modify the existing layers.
- Use different parameters for the layers.
- Use different parameters for the genetic algorithm (e.g. number of solution, number of generations, etc)
