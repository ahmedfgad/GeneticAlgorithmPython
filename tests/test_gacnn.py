import pygad.cnn
import pygad.gacnn
import pygad
import numpy

def test_gacnn_evolution():
    """Test pygad.gacnn with pygad.GA."""
    # Small dummy data
    data_inputs = numpy.random.uniform(0, 1, (4, 10, 10, 3))
    data_outputs = numpy.array([0, 1, 1, 0])

    input_layer = pygad.cnn.Input2D(input_shape=(10, 10, 3))
    conv_layer = pygad.cnn.Conv2D(num_filters=2,
                                   kernel_size=3,
                                   previous_layer=input_layer,
                                   activation_function="relu")
    flatten_layer = pygad.cnn.Flatten(previous_layer=conv_layer)
    dense_layer = pygad.cnn.Dense(num_neurons=2, 
                                   previous_layer=flatten_layer,
                                   activation_function="softmax")

    model = pygad.cnn.Model(last_layer=dense_layer,
                            epochs=1,
                            learning_rate=0.01)

    gacnn_instance = pygad.gacnn.GACNN(model=model,
                                     num_solutions=4)

    def fitness_func(ga_instance, solution, sol_idx):
        predictions = gacnn_instance.population_networks[sol_idx].predict(data_inputs=data_inputs)
        correct_predictions = numpy.where(predictions == data_outputs)[0].size
        solution_fitness = (correct_predictions/data_outputs.size)*100
        return solution_fitness

    def callback_generation(ga_instance):
        population_matrices = pygad.gacnn.population_as_matrices(population_networks=gacnn_instance.population_networks, 
                                                                population_vectors=ga_instance.population)
        gacnn_instance.update_population_trained_weights(population_trained_weights=population_matrices)

    initial_population = pygad.gacnn.population_as_vectors(population_networks=gacnn_instance.population_networks)

    ga_instance = pygad.GA(num_generations=2, 
                           num_parents_mating=2, 
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           on_generation=callback_generation,
                           suppress_warnings=True)

    ga_instance.run()
    assert ga_instance.run_completed
    assert ga_instance.generations_completed == 2
    
    print("test_gacnn_evolution passed.")

if __name__ == "__main__":
    test_gacnn_evolution()
    print("\nAll GACNN tests passed!")
