from ..nn import nn

def validate_network_parameters(num_neurons_input, 
                                num_neurons_output, 
                                num_neurons_hidden_layers, 
                                output_activation, 
                                hidden_activations, 
                                num_solutions=None):
    """
    Validating the parameters passed to initial_population_networks() in addition to creating a list of the name(s) of the activation function(s) for the hidden layer(s). 
    In case that the value passed to the 'hidden_activations' parameter is a string not a list, then a list is created by replicating the passed name a number of times equal to the number of hidden layers (i.e. the length of the 'num_neurons_hidden_layers' parameter).
    If an invalid parameter found, an exception is raised and the execution stops.

    The function accepts the same parameters passed to the constructor of the GANN class.

    num_neurons_input: Number of neurons in the input layer.
    num_neurons_output: Number of neurons in the output layer.
    num_neurons_hidden_layers: A list holding the number of neurons in the hidden layer(s).
    output_activation: The name of the activation function of the output layer.
    hidden_activations: The name(s) of the activation function(s) of the hidden layer(s).
    num_solutions: Number of solutions (i.e. networks) in the population which defaults to None. The reason why this function sets a default value to the `num_solutions` parameter is differentiating whether a population of networks or a single network is to be created. If `None`, then a single network will be created. If not `None`, then a population of networks is to be created.
    
    Returns a list holding the name(s) of the activation function(s) for the hidden layer(s). 
    """

    # Validating the number of solutions within the population.
    if not (num_solutions is None):
        if num_solutions < 2:
            raise ValueError("num_solutions: The number of solutions within the population must be at least 2. The current value is {num_solutions}.".format(num_solutions=num_solutions))
        
    # Validating the number of neurons in the input layer.
    if num_neurons_input is int and num_neurons_input <= 0:
        raise ValueError("num_neurons_input: The number of neurons in the input layer must be > 0.")
    
    # Validating the number of neurons in the output layer.
    if num_neurons_output is int and num_neurons_output <= 0:
        raise ValueError("num_neurons_output: The number of neurons in the output layer must be > 0.")
    
    # Validating the type of the 'num_neurons_hidden_layers' parameter which is expected to be list or tuple.
    if not (type(num_neurons_hidden_layers) in [list, tuple]):
        raise TypeError("num_neurons_hidden_layers: A list or a tuple is expected but {hidden_layers_neurons_type} found.".format(hidden_layers_neurons_type=type(num_neurons_hidden_layers)))
    
    # Frequently used error messages.
    unexpected_output_activation_value = "Output activation function: The activation function of the output layer is passed as a string not {activation_type}."
    unexpected_activation_value = "Activation function: The supported values for the activation function are {supported_activations} but an unexpected value is found:\n{activations}"
    unexpected_activation_type = "Activation Function: A list, tuple, or a string is expected but {activations_type} found."
    length_mismatch = "Hidden activation functions: When passing the activation function(s) as a list or a tuple, its length must match the length of the 'num_neurons_hidden_layers' parameter but a mismatch is found:\n{mismatched_lengths}"

    # A list of the names of the supported activation functions.
    supported_activations = ["sigmoid", "relu", "softmax", "None"]

    # Validating the output layer activation function.
    if not (type(output_activation) is str):
        raise ValueError(unexpected_output_activation_value.format(activation_type=type(output_activation)))
    if not (output_activation in supported_activations): #activation_type
        raise ValueError(unexpected_activation_value.format(activations=output_activation, supported_activations=supported_activations))
    
    # Number of hidden layers.
    num_hidden_layers = len(num_neurons_hidden_layers)
    if num_hidden_layers > 1: # In case there are more than 1 hidden layer.
        if type(hidden_activations) in [list, tuple]:
            num_activations = len(hidden_activations)
            if num_activations != num_hidden_layers:
                raise ValueError(length_mismatch.format(mismatched_lengths="{num_activations} != {num_layers}".format(num_layers=num_hidden_layers, num_activations=num_activations)))
        elif type(hidden_activations) is str:
            if hidden_activations in supported_activations:
                hidden_activations = [hidden_activations]*num_hidden_layers
            else:
                raise ValueError(unexpected_activation_value.format(supported_activations=supported_activations, activations=hidden_activations))
        else:
            raise TypeError(unexpected_activation_type.format(activations_type=type(hidden_activations)))
    elif num_hidden_layers == 1:  # In case there is only 1 hidden layer.
        if (type(hidden_activations) in [list, tuple]):
            if len(hidden_activations) != 1:
                raise ValueError(length_mismatch.format(mismatched_lengths="{num_activations} != {num_layers}".format(num_layers=num_hidden_layers, num_activations=len(hidden_activations))))
        elif type(hidden_activations) is str:
            if not (hidden_activations in supported_activations):
                raise ValueError(unexpected_activation_value.format(supported_activations=supported_activations, activations=hidden_activations))
            else:
                hidden_activations = [hidden_activations]
        else:
            raise TypeError(unexpected_activation_type.format(activations_type=type(hidden_activations)))
    else: # In case there are no hidden layers (num_hidden_layers == 0)
        print("WARNING: There are no hidden layers however a value is assigned to the parameter 'hidden_activations'. It will be reset to [].".format(hidden_activations=hidden_activations))
        hidden_activations = []
    
    # If the value passed to the 'hidden_activations' parameter is actually a list, then its elements are checked to make sure the listed name(s) of the activation function(s) are supported.
    for act in hidden_activations:
        if not (act in supported_activations):
            raise ValueError(unexpected_activation_value.format(supported_activations=supported_activations, activations=act))

    return hidden_activations

def create_network(num_neurons_input, 
                   num_neurons_output, 
                   num_neurons_hidden_layers=[], 
                   output_activation="softmax", 
                   hidden_activations="relu", 
                   parameters_validated=False):
    """
    Creates a neural network as a linked list between the input, hidden, and output layers where the layer at index N (which is the last/output layer) references the layer at index N-1 (which is a hidden layer) using its previous_layer attribute. The input layer does not reference any layer because it is the last layer in the linked list.

    In addition to the parameters_validated parameter, this function accepts the same parameters passed to the constructor of the gann.GANN class except for the num_solutions parameter because only a single network is created out of the create_network() function.

    num_neurons_input: Number of neurons in the input layer.
    num_neurons_output: Number of neurons in the output layer.
    num_neurons_hidden_layers=[]: A list holding the number of neurons in the hidden layer(s). If empty [], then no hidden layers are used. For each int value it holds, then a hidden layer is created with number of hidden neurons specified by the corresponding int value. For example, num_neurons_hidden_layers=[10] creates a single hidden layer with 10 neurons. num_neurons_hidden_layers=[10, 5] creates 2 hidden layers with 10 neurons for the first and 5 neurons for the second hidden layer.
    output_activation="softmax": The name of the activation function of the output layer which defaults to "softmax".
    hidden_activations="relu": The name(s) of the activation function(s) of the hidden layer(s). It defaults to "relu". If passed as a string, this means the specified activation function will be used across all the hidden layers. If passed as a list, then it must has the same length as the length of the num_neurons_hidden_layers list. An exception is raised if there lengths are different. When hidden_activations is a list, a one-to-one mapping between the num_neurons_hidden_layers and hidden_activations lists occurs.
    parameters_validated=False: If False, then the parameters are not validated and a call to the validate_network_parameters() function is made.

    Returns the reference to the last layer in the network architecture which is the output layer. Based on such reference, all network layer can be fetched.    
    """
    
    # When parameters_validated is False, then the parameters are not yet validated and a call to validate_network_parameters() is required.
    if parameters_validated == False:
        # Validating the passed parameters before creating the network.
        hidden_activations = validate_network_parameters(num_neurons_input=num_neurons_input,
                                                         num_neurons_output=num_neurons_output,
                                                         num_neurons_hidden_layers=num_neurons_hidden_layers,
                                                         output_activation=output_activation,
                                                         hidden_activations=hidden_activations)

    # Creating the input layer as an instance of the nn.InputLayer class.
    input_layer = nn.InputLayer(num_neurons_input)
    
    if len(num_neurons_hidden_layers) > 0:
        # If there are hidden layers, then the first hidden layer is connected to the input layer.
        hidden_layer = nn.DenseLayer(num_neurons=num_neurons_hidden_layers.pop(0), 
                                     previous_layer=input_layer, 
                                     activation_function=hidden_activations.pop(0))
        # For the other hidden layers, each hidden layer is connected to its preceding hidden layer.
        for hidden_layer_idx in range(len(num_neurons_hidden_layers)):
            hidden_layer = nn.DenseLayer(num_neurons=num_neurons_hidden_layers.pop(0), 
                                         previous_layer=hidden_layer, 
                                         activation_function=hidden_activations.pop(0))

        # The last hidden layer is connected to the output layer.
        # The output layer is created as an instance of the nn.DenseLayer class.
        output_layer = nn.DenseLayer(num_neurons=num_neurons_output, 
                                     previous_layer=hidden_layer, 
                                     activation_function=output_activation)

    # If there are no hidden layers, then the output layer is connected directly to the input layer.
    elif len(num_neurons_hidden_layers) == 0:
        # The output layer is created as an instance of the nn.DenseLayer class.
        output_layer = nn.DenseLayer(num_neurons=num_neurons_output, 
                                     previous_layer=input_layer,
                                     activation_function=output_activation)

    # Returning the reference to the last layer in the network architecture which is the output layer. Based on such reference, all network layer can be fetched.
    return output_layer

def population_as_vectors(population_networks):
    """
    Accepts the population as networks and returns a list holding all weights of the layers of each solution (i.e. network) in the population as a vector. 
    If the population has 6 solutions (i.e. networks), this function accepts references to such networks and returns a list with 6 vectors, one for each network (i.e. solution). Each vector holds the weights for all layers for a single network.
    
    population_networks: A list holding references to the output (last) layers of the neural networks used in the population. 
    
    Returns a list holding the weights vectors for all solutions (i.e. networks).
    """
    population_vectors = []
    for solution in population_networks:
        # Converting the weights of single layer from the current network (i.e. solution) to a vector.
        solution_weights_vector = nn.layers_weights_as_vector(solution)
        # Appending the weights vector of the current layer of a network (i.e. solution) to the weights of the previous layers of the same network (i.e. solution).
        population_vectors.append(solution_weights_vector)

    return population_vectors

def population_as_matrices(population_networks, population_vectors):
    """
    Accepts the population as both networks and weights vectors and returns the weights of all layers of each solution (i.e. network) in the population as a matrix.
    If the population has 6 solutions (i.e. networks), this function returns a list with 6 matrices, one for each network holding its weights for all layers.

    population_networks: A list holding references to the output (last) layers of the neural networks used in the population. 
    population_vectors: A list holding the weights of all networks as vectors. Such vectors are to be converted into matrices.

    Returns a list holding the weights matrices for all solutions (i.e. networks).
    """
    population_matrices = []
    for solution, solution_weights_vector in zip(population_networks, population_vectors):
        # Converting the weights of single layer from the current network (i.e. solution) from a vector to a matrix.
        solution_weights_matrix = nn.layers_weights_as_matrix(solution, solution_weights_vector)
        # Appending the weights matrix of the current layer of a network (i.e. solution) to the weights of the previous layers of the same network (i.e. solution).
        population_matrices.append(solution_weights_matrix)

    return population_matrices

class GANN:
    def create_population(self):
        """
        Creates the initial population of the genetic algorithm as a list of neural networks (i.e. solutions). Each element in the list holds a reference to the last (i.e. output) layer for the network. The method does not accept any parameter and it accesses all the required details from the `GANN` instance.

        The method returns the list holding the references to the networks.
        """

        population_networks = []
        for solution in range(self.num_solutions):
            # Creating a network (i.e. solution) in the population. A network or a solution can be used interchangeably.
            # .copy() is so important to avoid modification in the original vale passed to the 'num_neurons_hidden_layers' and 'hidden_activations' parameters.
            network = create_network(num_neurons_input=self.num_neurons_input,
                                     num_neurons_output=self.num_neurons_output,
                                     num_neurons_hidden_layers=self.num_neurons_hidden_layers.copy(),
                                     output_activation=self.output_activation, 
                                     hidden_activations=self.hidden_activations.copy(),
                                     parameters_validated=True)

            # Appending the created network to the list of population networks.
            population_networks.append(network)

        return population_networks
    
    def __init__(self, 
                 num_solutions, 
                 num_neurons_input, 
                 num_neurons_output, 
                 num_neurons_hidden_layers=[], 
                 output_activation="softmax", 
                 hidden_activations="relu"):
        """
        Creates an instance of the GANN class for training a neural network using the genetic algorithm.
        The constructor of the GANN class creates an initial population of multiple neural networks using the create_population() method. 
        The population returned holds references to the last (i.e. output) layers of all created networks.
        Besides creating the initial population, the passed parameters are vaidated using the validate_network_parameters() method.
    
        num_solutions: Number of neural networks (i.e. solutions) in the population. Based on the value passed to this parameter, a number of identical neural networks are created where their parameters are optimized using the genetic algorithm.
        num_neurons_input: Number of neurons in the input layer.
        num_neurons_output: Number of neurons in the output layer.
        num_neurons_hidden_layers=[]: A list holding the number of neurons in the hidden layer(s). If empty [], then no hidden layers are used. For each int value it holds, then a hidden layer is created with number of hidden neurons specified by the corresponding int value. For example, num_neurons_hidden_layers=[10] creates a single hidden layer with 10 neurons. num_neurons_hidden_layers=[10, 5] creates 2 hidden layers with 10 neurons for the first and 5 neurons for the second hidden layer.
        output_activation="softmax": The name of the activation function of the output layer which defaults to "softmax".
        hidden_activations="relu": The name(s) of the activation function(s) of the hidden layer(s). It defaults to "relu". If passed as a string, this means the specified activation function will be used across all the hidden layers. If passed as a list, then it must has the same length as the length of the num_neurons_hidden_layers list. An exception is raised if there lengths are different. When hidden_activations is a list, a one-to-one mapping between the num_neurons_hidden_layers and hidden_activations lists occurs.
        """

        self.parameters_validated = False # If True, then the parameters  passed to the GANN class constructor are valid.

        # Validating the passed parameters before building the initial population.
        hidden_activations = validate_network_parameters(num_solutions=num_solutions,
                                                         num_neurons_input=num_neurons_input,
                                                         num_neurons_output=num_neurons_output,
                                                         num_neurons_hidden_layers=num_neurons_hidden_layers,
                                                         output_activation=output_activation,
                                                         hidden_activations=hidden_activations)

        self.num_solutions = num_solutions
        self.num_neurons_input = num_neurons_input
        self.num_neurons_output = num_neurons_output
        self.num_neurons_hidden_layers = num_neurons_hidden_layers
        self.output_activation = output_activation
        self.hidden_activations = hidden_activations
        self.parameters_validated = True

        # After the parameters are validated, the initial population is created.
        self.population_networks = self.create_population() # A list holding references to all the solutions (i.e. neural networks) used in the population.

    def update_population_trained_weights(self, population_trained_weights):
        """
        The `update_population_trained_weights()` method updates the `trained_weights` attribute of each network (check the [documentation of the `pygad.nn.DenseLayer` class](https://github.com/ahmedfgad/NumPyANN#nndenselayer-class) for more information) according to the weights passed in the `population_trained_weights` parameter.

        population_trained_weights: A list holding the trained weights of all networks as matrices. Such matrices are to be assigned to the 'trained_weights' attribute of all layers of all networks.
        """
        idx = 0
        # Fetches all layers weights matrices for a single solution (i.e. network)
        for solution in self.population_networks:
            # Calling the nn.update_layers_trained_weights() function for updating the 'trained_weights' attribute for all layers in the current solution (i.e. network).
            nn.update_layers_trained_weights(last_layer=solution, 
                                             final_weights=population_trained_weights[idx])
            idx = idx + 1
