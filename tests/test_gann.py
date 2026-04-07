import pygad.gann
import pygad.nn
import numpy

def test_gann_regression():
    """Test GANN for a simple regression problem."""
    # Data
    data_inputs = numpy.array([[0.02, 0.1, 0.15],
                               [0.7, 0.6, 0.8],
                               [1.5, 1.2, 1.7],
                               [3.2, 2.9, 3.1]])
    data_outputs = numpy.array([0.1, 0.6, 1.3, 2.5])

    # GANN architecture
    num_inputs = data_inputs.shape[1]
    num_classes = 1 # Regression

    gann_instance = pygad.gann.GANN(num_solutions=10,
                                     num_neurons_input=num_inputs,
                                     num_neurons_hidden_layers=[5],
                                     num_neurons_output=num_classes,
                                     hidden_activations="relu",
                                     output_activation="None")

    # The number of genes is the total number of weights in the network.
    # We can get it by converting the weights of any network in the population into a vector.
    num_genes = len(pygad.nn.layers_weights_as_vector(last_layer=gann_instance.population_networks[0]))

    def fitness_func(ga_instance, solution, solution_idx):
        # Update the weights of the network associated with the current solution.
        # GANN.update_population_trained_weights expects weights for ALL solutions.
        # To avoid updating all, we can update just the one we need.
        
        # However, for simplicity and to test GANN's intended flow:
        population_matrices = pygad.gann.population_as_matrices(num_networks=ga_instance.sol_per_pop,
                                                               population_vectors=ga_instance.population)
        gann_instance.update_population_trained_weights(population_trained_weights=population_matrices)

        predictions = pygad.nn.predict(last_layer=gann_instance.population_networks[solution_idx],
                                       data_inputs=data_inputs)
        
        # Mean Absolute Error
        abs_error = numpy.mean(numpy.abs(predictions.flatten() - data_outputs)) + 0.00000001
        fitness = 1.0 / abs_error
        return fitness

    ga_instance = pygad.GA(num_generations=5,
                           num_parents_mating=4,
                           fitness_func=fitness_func,
                           sol_per_pop=10,
                           num_genes=num_genes,
                           random_seed=42,
                           suppress_warnings=True)

    ga_instance.run()
    assert ga_instance.run_completed
    print("test_gann_regression passed.")

def test_nn_direct_usage():
    """Test pygad.nn layers directly."""
    input_layer = pygad.nn.InputLayer(num_inputs=3)
    dense_layer = pygad.nn.DenseLayer(num_neurons=2, previous_layer=input_layer, activation_function="relu")
    output_layer = pygad.nn.DenseLayer(num_neurons=1, previous_layer=dense_layer, activation_function="sigmoid")
    
    data_inputs = numpy.array([[0.1, 0.2, 0.3]])
    predictions = pygad.nn.predict(last_layer=output_layer, data_inputs=data_inputs)
    
    assert predictions.shape == (1, 1)
    assert 0 <= predictions[0, 0] <= 1
    print("test_nn_direct_usage passed.")

if __name__ == "__main__":
    test_gann_regression()
    test_nn_direct_usage()
    print("\nAll GANN/NN tests passed!")
