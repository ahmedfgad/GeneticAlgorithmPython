import tensorflow as tf
import tensorflow.keras
import pygad.kerasga
import pygad
import numpy

def fitness_func(ga_instanse, solution, sol_idx):
    global train_data, data_outputs, keras_ga, model

    predictions = pygad.kerasga.predict(model=model,
                                        solution=solution,
                                        data=train_data)

    cce = tensorflow.keras.losses.CategoricalCrossentropy()
    solution_fitness = 1.0 / (cce(data_outputs, predictions).numpy() + 0.00000001)

    return solution_fitness

def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")

# The dataset path.
dataset_path = r'../data/Skin_Cancer_Dataset' 

num_classes = 2
img_size = 224

# Create a simple CNN. This does not gurantee high classification accuracy.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(img_size, img_size, 3)))
model.add(tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

# Create an instance of the pygad.kerasga.KerasGA class to build the initial population.
keras_ga = pygad.kerasga.KerasGA(model=model,
                                 num_solutions=10)

train_data = tf.keras.utils.image_dataset_from_directory(
    directory=dataset_path,
    image_size=(img_size, img_size),
    label_mode="categorical",
    batch_size=32
)

# Get the dataset labels.
# train_data.class_indices
data_outputs =  numpy.array([])
for x, y in train_data:
    data_outputs = numpy.concatenate([data_outputs, numpy.argmax(y.numpy(), axis=-1)])
data_outputs = tf.keras.utils.to_categorical(data_outputs)

# Check the documentation for more information about the parameters: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
initial_population = keras_ga.population_weights # Initial population of network weights.

# Create an instance of the pygad.GA class
ga_instance = pygad.GA(num_generations=10, 
                       num_parents_mating=5, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=on_generation)

# Start the genetic algorithm evolution.
ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

predictions = pygad.kerasga.predict(model=model,
                                    solution=solution,
                                    data=train_data)
# print("Predictions : \n", predictions)

# Calculate the categorical crossentropy for the trained model.
cce = tensorflow.keras.losses.CategoricalCrossentropy()
print(f"Categorical Crossentropy : {cce(data_outputs, predictions).numpy()}")

# Calculate the classification accuracy for the trained model.
ca = tensorflow.keras.metrics.CategoricalAccuracy()
ca.update_state(data_outputs, predictions)
accuracy = ca.result().numpy()
print(f"Accuracy : {accuracy}")

# model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

# _ = model.fit(x, y, verbose=0)
# r = model.predict(data_inputs)
