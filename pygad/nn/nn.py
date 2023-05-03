import numpy
import functools

"""
This project creates a neural network where the architecture has input and dense layers only. More layers will be added in the future. 
The project only implements the forward pass of a neural network and no training algorithm is used.
For training a neural network using the genetic algorithm, check this project (https://github.com/ahmedfgad/NeuralGenetic) in which the genetic algorithm is used for training the network.
Feel free to leave an issue in this project (https://github.com/ahmedfgad/NumPyANN) in case something is not working properly or to ask for questions. I am also available for e-mails at ahmed.f.gad@gmail.com
"""

def layers_weights(last_layer, initial=True):
    """
    Creates a list holding the weights of all layers in the neural network.

    last_layer: A reference to the last (output) layer in the network architecture.
    initial: When True, the function returns the initial weights of the layers. When False, the trained weights of the layers are returned. The initial weights are only needed before network training starts. The trained weights are needed to predict the network outputs.

    Returns a list (network_weights) holding the weights of the layers.
    """
    network_weights = []

    layer = last_layer
    while "previous_layer" in layer.__init__.__code__.co_varnames:
        # If the 'initial' parameter is True, append the initial weights. Otherwise, append the trained weights.
        if initial == True:
            network_weights.append(layer.initial_weights)
        elif initial == False:
            network_weights.append(layer.trained_weights)
        else:
            raise ValueError("Unexpected value to the 'initial' parameter: {initial}.".format(initial=initial))

        # Go to the previous layer.
        layer = layer.previous_layer

    # If the first layer in the network is not an input layer (i.e. an instance of the InputLayer class), raise an error.
    if not (type(layer) is InputLayer):
        raise TypeError("The first layer in the network architecture must be an input layer.")

    # Currently, the weights of the layers are in the reverse order. In other words, the weights of the first layer are at the last index of the 'network_weights' list while the weights of the last layer are at the first index.
    # Reversing the 'network_weights' list to order the layers' weights according to their location in the network architecture (i.e. the weights of the first layer appears at index 0 of the list).
    network_weights.reverse()
    return network_weights

def layers_weights_as_vector(last_layer, initial=True):
    """
    Creates a list holding the weights of each layer in the network as a vector.

    last_layer: A reference to the last (output) layer in the network architecture.
    initial: When True, the function returns the initial weights of the layers. When False, the trained weights of the layers are returned. The initial weights are only needed before network training starts. The trained weights are needed to predict the network outputs.
    
    Returns a list (network_weights) holding the weights of the layers as a vector.
    """
    network_weights = []

    layer = last_layer
    while "previous_layer" in layer.__init__.__code__.co_varnames:
        # If the 'initial' parameter is True, append the initial weights. Otherwise, append the trained weights.
        if initial == True:
            vector = numpy.reshape(layer.initial_weights, newshape=(layer.initial_weights.size))
#            vector = DenseLayer.to_vector(matrix=layer.initial_weights)
            network_weights.extend(vector)
        elif initial == False:
            vector = numpy.reshape(layer.trained_weights, newshape=(layer.trained_weights.size))
#            vector = DenseLayer.to_vector(array=layer.trained_weights)
            network_weights.extend(vector)
        else:
            raise ValueError("Unexpected value to the 'initial' parameter: {initial}.".format(initial=initial))

        # Go to the previous layer.
        layer = layer.previous_layer

    # If the first layer in the network is not an input layer (i.e. an instance of the InputLayer class), raise an error.
    if not (type(layer) is InputLayer):
        raise TypeError("The first layer in the network architecture must be an input layer.")

    # Currently, the weights of the layers are in the reverse order. In other words, the weights of the first layer are at the last index of the 'network_weights' list while the weights of the last layer are at the first index.
    # Reversing the 'network_weights' list to order the layers' weights according to their location in the network architecture (i.e. the weights of the first layer appears at index 0 of the list).
    network_weights.reverse()
    return numpy.array(network_weights)

def layers_weights_as_matrix(last_layer, vector_weights):
    """
    Converts the network weights from vectors to matrices.

    last_layer: A reference to the last (output) layer in the network architecture.
    vector_weights: The network weights as vectors where the weights of each layer form a single vector.

    Returns a list (network_weights) holding the weights of the layers as matrices.
    """
    network_weights = []

    start = 0
    layer = last_layer
    vector_weights = vector_weights[::-1]
    while "previous_layer" in layer.__init__.__code__.co_varnames:
        layer_weights_shape = layer.initial_weights.shape
        layer_weights_size = layer.initial_weights.size

        weights_vector=vector_weights[start:start + layer_weights_size]
#        matrix = DenseLayer.to_array(vector=weights_vector, shape=layer_weights_shape)
        matrix = numpy.reshape(weights_vector, newshape=(layer_weights_shape))
        network_weights.append(matrix)

        start = start + layer_weights_size

        # Go to the previous layer.
        layer = layer.previous_layer

    # If the first layer in the network is not an input layer (i.e. an instance of the InputLayer class), raise an error.
    if not (type(layer) is InputLayer):
        raise TypeError("The first layer in the network architecture must be an input layer.")

    # Currently, the weights of the layers are in the reverse order. In other words, the weights of the first layer are at the last index of the 'network_weights' list while the weights of the last layer are at the first index.
    # Reversing the 'network_weights' list to order the layers' weights according to their location in the network architecture (i.e. the weights of the first layer appears at index 0 of the list).
    network_weights.reverse()
    return network_weights

def layers_activations(last_layer):
    """
    Creates a list holding the activation functions of all layers in the network.
    
    last_layer: A reference to the last (output) layer in the network architecture.
    
    Returns a list (activations) holding the activation functions of the layers.
    """
    activations = []

    layer = last_layer
    while "previous_layer" in layer.__init__.__code__.co_varnames:
        activations.append(layer.activation_function)

        # Go to the previous layer.
        layer = layer.previous_layer

    if not (type(layer) is InputLayer):
        raise TypeError("The first layer in the network architecture must be an input layer.")

    # Currently, the activations of layers are in the reverse order. In other words, the activation function of the first layer are at the last index of the 'activations' list while the activation function of the last layer are at the first index.
    # Reversing the 'activations' list to order the layers' weights according to their location in the network architecture (i.e. the activation function of the first layer appears at index 0 of the list).
    activations.reverse()
    return activations

def sigmoid(sop):

    """
    Applies the sigmoid function.

    sop: The input to which the sigmoid function is applied.

    Returns the result of the sigmoid function.
    """

    if type(sop) in [list, tuple]:
        sop = numpy.array(sop)

    return 1.0 / (1 + numpy.exp(-1 * sop))

def relu(sop):

    """
    Applies the rectified linear unit (ReLU) function.

    sop: The input to which the relu function is applied.

    Returns the result of the ReLU function.
    """

    if not (type(sop) in [list, tuple, numpy.ndarray]):
        if sop < 0:
            return 0
        else:
            return sop
    elif type(sop) in [list, tuple]:
        sop = numpy.array(sop)

    result = sop
    result[sop < 0] = 0

    return result

def softmax(layer_outputs):

    """
    Applies the sotmax function.

    sop: The input to which the softmax function is applied.

    Returns the result of the softmax function.
    """
    return layer_outputs / (numpy.sum(layer_outputs) + 0.000001)

def train(num_epochs, 
          last_layer, 
          data_inputs, 
          data_outputs,
          problem_type="classification",
          learning_rate=0.01):
    """
    Trains the neural network.
    
    num_epochs: Number of epochs.
    last_layer: Reference to the last (output) layer in the network architecture.
    data_inputs: Data features.
    data_outputs: Data outputs.
    problem_type: Can be either classification or regression to define the problem type.
    learning_rate: Learning rate which defaults to 0.01.
    """
    
    if not (problem_type in ["classification", "regression"]):
        raise ValueError("The value of the problem_type parameter can be either classification or regression but {problem_type_val} found.".format(problem_type_val=problem_type))
    
    # To fetch the initial weights of the layer, the 'initial' argument is set to True.
    weights = layers_weights(last_layer, initial=True)
    activations = layers_activations(last_layer)
    
    network_error = 0
    for epoch in range(num_epochs):
        print("Epoch ", epoch)
        for sample_idx in range(data_inputs.shape[0]):
            r1 = data_inputs[sample_idx, :]
            for idx in range(len(weights) - 1):
                curr_weights = weights[idx]
                r1 = numpy.matmul(r1, curr_weights)
                if activations[idx] == "relu":
                    r1 = relu(r1)
                elif activations[idx] == "sigmoid":
                    r1 = sigmoid(r1)
                elif activations[idx] == "softmax":
                    r1 = softmax(r1)
                elif activations[idx] == None:
                    pass

            curr_weights = weights[-1]
            r1 = numpy.matmul(r1, curr_weights)

            if problem_type == "classification":
                prediction = numpy.where(r1 == numpy.max(r1))[0][0]
            else:
                prediction = r1

            network_error = network_error + numpy.mean(numpy.abs((prediction - data_outputs[sample_idx])))

        # Updating the network weights once after completing an epoch (i.e. passing through all the samples).
        weights = update_weights(weights=weights,
                                 network_error=network_error,
                                 learning_rate=learning_rate)

    # Initially, the 'trained_weights' attribute of the layers are set to None. After the is trained, the 'trained_weights' attribute is updated by the trained weights using the update_layers_trained_weights() function.
    update_layers_trained_weights(last_layer, weights)

def update_weights(weights, network_error, learning_rate):
    """
    Updates the network weights using the learning rate only. 
    The purpose of this project is to only apply the forward pass of training a neural network. Thus, there is no optimization algorithm is used like the gradient descent.
    For optimizing the neural network, check this project (https://github.com/ahmedfgad/NeuralGenetic) in which the genetic algorithm is used for training the network.
    
    weights: The current weights of the network.
    network_error: The network error.
    learning_rate: The learning rate.

    It returns the new weights.
    """
    # weights = numpy.array(weights)
    for layer_idx in range(len(weights)):
        weights[layer_idx] = network_error * learning_rate * weights[layer_idx]

    return weights

def update_layers_trained_weights(last_layer, final_weights):
    """
    After the network weights are trained, the 'trained_weights' attribute of each layer is updated by the weights calculated after passing all the epochs (such weights are passed in the 'final_weights' parameter).
    By just passing a reference to the last layer in the network (i.e. output layer) in addition to the final weights, this function updates the 'trained_weights' attribute of all layers.

    last_layer: A reference to the last (output) layer in the network architecture.
    final_weights: An array of weights of all layers in the network after passing through all the epochs.
    """
    layer = last_layer
    layer_idx = len(final_weights) - 1
    while "previous_layer" in layer.__init__.__code__.co_varnames:
        layer.trained_weights = final_weights[layer_idx]

        layer_idx = layer_idx - 1
        # Go to the previous layer.
        layer = layer.previous_layer

def predict(last_layer, data_inputs, problem_type="classification"):
    """
    Uses the trained weights for predicting the samples' outputs.

    last_layer: A reference to the last (output) layer in the network architecture.
    data_inputs: Data features.
    problem_type: Can be either classification or regression to define the problem type.

    Returns the predictions of all samples.
    """
    if not (problem_type in ["classification", "regression"]):
        raise ValueError("The value of the problem_type parameter can be either classification or regression but {problem_type_val} found.".format(problem_type_val=problem_type))
    
    # To fetch the trained weights of the layer, the 'initial' argument is set to False.
    weights = layers_weights(last_layer, initial=False)
    activations = layers_activations(last_layer)

    if len(weights) != len(activations):
        raise TypeError("The length of layers {num_layers} is not equal to the number of activations functions {num_activations} and they must be equal.".format(num_layers=len(weights), num_activations=len(activations)))

    predictions = []
    for sample_idx in range(data_inputs.shape[0]):
        r1 = data_inputs[sample_idx, :]
        for curr_weights, activation in zip(weights, activations):
            r1 = numpy.matmul(r1, curr_weights)
            if activation == "relu":
                r1 = relu(r1)
            elif activation == "sigmoid":
                r1 = sigmoid(r1)
            elif activation == "softmax":
                r1 = softmax(r1)
            elif activation == None:
                pass

        if problem_type == "classification":
            prediction = numpy.where(r1 == numpy.max(r1))[0][0]
        else:
            prediction = r1

        predictions.append(prediction)

    return predictions

def to_vector(array):
    """
    Converts a passed NumPy array (of any dimensionality) to its `array`  parameter into a 1D vector and returns the vector.
    
    array: The NumPy array to be converted into a 1D vector.

    Returns the array after being reshaped into a NumPy 1D vector.

    Example: weights_vector = nn.DenseLayer.to_vector(array=array)
    """
    if not (type(array) is numpy.ndarray):
        raise TypeError("An input of type numpy.ndarray is expected but an input of type {in_type} found.".format(in_type=type(array)))
    return numpy.reshape(array, newshape=(array.size))

def to_array(vector, shape):
    """
    Converts a passed vector to its `vector`  parameter into a NumPy array and returns the array.

    vector: The 1D vector to be converted into an array.
    shape: The target shape of the array.

    Returns the NumPy 1D vector after being reshaped into an array.

    Example: weights_matrix = nn.DenseLayer.to_array(vector=vector, shape=shape)
    """
    if not (type(vector) is numpy.ndarray):
        raise TypeError("An input of type numpy.ndarray is expected but an input of type {in_type} found.".format(in_type=type(vector)))
    if vector.ndim > 1:
        raise ValueError("A 1D NumPy array is expected but an array of {ndim} dimensions found.".format(ndim=vector.ndim))
    if vector.size != functools.reduce(lambda x,y:x*y, shape, 1): # (operator.mul == lambda x,y:x*y
        raise ValueError("Mismatch between the vector length and the array shape. A vector of length {vector_length} cannot be converted into a array of shape ({array_shape}).".format(vector_length=vector.size, array_shape=shape))
    return numpy.reshape(vector, newshape=shape)

class InputLayer:
    """
    Implementing the input layer of a neural network.
    """
    def __init__(self, num_inputs):
        if num_inputs <= 0:
            raise ValueError("Number of input neurons cannot be <= 0. Please pass a valid value to the 'num_inputs' parameter.")
        # The number of neurons in the input layer.
        self.num_neurons = num_inputs

class DenseLayer:
    """
    Implementing the input dense (fully connected) layer of a neural network.
    """
    def __init__(self, num_neurons, previous_layer, activation_function="sigmoid"):
        if num_neurons <= 0:
            raise ValueError("Number of neurons cannot be <= 0. Please pass a valid value to the 'num_neurons' parameter.")
        # Number of neurons in the dense layer.
        self.num_neurons = num_neurons

        supported_activation_functions = ("sigmoid", "relu", "softmax", "None")
        if not (activation_function in supported_activation_functions):
            raise ValueError("The specified activation function '{activation_function}' is not among the supported activation functions {supported_activation_functions}. Please use one of the supported functions.".format(activation_function=activation_function, supported_activation_functions=supported_activation_functions))
        self.activation_function = activation_function

        if previous_layer is None:
            raise TypeError("The previous layer cannot be of Type 'None'. Please pass a valid layer to the 'previous_layer' parameter.")
        # A reference to the layer that preceeds the current layer in the network architecture.
        self.previous_layer = previous_layer

        # Initializing the weights of the layer.
        self.initial_weights = numpy.random.uniform(low=-0.1,
                                                    high=0.1,
                                                    size=(previous_layer.num_neurons, num_neurons))

        # The trained weights of the layer. Only assigned a value after the network is trained (i.e. the train() function completes).
        # Just initialized to be equal to the initial weights
        self.trained_weights = self.initial_weights.copy()