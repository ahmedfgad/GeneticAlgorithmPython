import numpy
import pygad
import pygad.kerasga
import tensorflow.keras

def test_kerasga_evolution():
    """Test pygad.kerasga with pygad.GA."""

    # XOR data
    data_inputs = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    data_outputs = numpy.array([[1, 0], [0, 1], [0, 1], [1, 0]]) # One-hot encoded

    input_layer = tensorflow.keras.layers.Input(shape=(2,))
    dense_layer = tensorflow.keras.layers.Dense(4, activation="relu")(input_layer)
    output_layer = tensorflow.keras.layers.Dense(2, activation="softmax")(dense_layer)

    model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)

    keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)

    def fitness_func(ga_instance, solution, solution_idx):
        model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                     weights_vector=solution)
        model.set_weights(weights=model_weights_matrix)
        predictions = model.predict(data_inputs, verbose=0)
        
        cce = tensorflow.keras.losses.CategoricalCrossentropy()
        loss = cce(data_outputs, predictions).numpy()
        fitness = 1.0 / (loss + 0.00000001)
        return fitness

    ga_instance = pygad.GA(num_generations=2,
                           num_parents_mating=5,
                           initial_population=keras_ga.population_weights,
                           fitness_func=fitness_func,
                           suppress_warnings=True)

    ga_instance.run()
    assert ga_instance.run_completed
    assert ga_instance.generations_completed == 2
    
    print("test_kerasga_evolution passed.")

if __name__ == "__main__":
    test_kerasga_evolution()
    print("\nAll KerasGA tests passed!")
