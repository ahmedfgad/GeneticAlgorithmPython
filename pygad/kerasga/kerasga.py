import copy
import numpy
import tensorflow.keras

def model_weights_as_vector(model):
    """
    Reshapes the Keras model weight as a vector.

    Parameters
    ----------
    model : TYPE
        The Keras model.

    Returns
    -------
    TYPE
        The weights as a 1D vector.

    """
    weights_vector = []

    for layer in model.layers: # model.get_weights():
        if layer.trainable:
            layer_weights = layer.get_weights()
            for l_weights in layer_weights:
                vector = numpy.reshape(l_weights, newshape=(l_weights.size))
                weights_vector.extend(vector)

    return numpy.array(weights_vector)

def model_weights_as_matrix(model, weights_vector):
    """
    Reshapes the PyGAD 1D solution as a Keras weight matrix.

    Parameters
    ----------
    model : TYPE
        The Keras model.
    weights_vector : TYPE
        The PyGAD solution as a 1D vector.

    Returns
    -------
    weights_matrix : TYPE
        The Keras weights as a matrix.

    """
    weights_matrix = []

    start = 0
    for layer_idx, layer in enumerate(model.layers): # model.get_weights():
    # for w_matrix in model.get_weights():
        layer_weights = layer.get_weights()
        if layer.trainable:
            for l_weights in layer_weights:
                layer_weights_shape = l_weights.shape
                layer_weights_size = l_weights.size
        
                layer_weights_vector = weights_vector[start:start + layer_weights_size]
                layer_weights_matrix = numpy.reshape(layer_weights_vector, newshape=(layer_weights_shape))
                weights_matrix.append(layer_weights_matrix)
        
                start = start + layer_weights_size
        else:
            for l_weights in layer_weights:
                weights_matrix.append(l_weights)

    return weights_matrix

def predict(model, 
            solution, 
            data, 
            batch_size=None,
            verbose=0,
            steps=None):
    """
    Use the PyGAD's solution to make predictions using the Keras model.

    Parameters
    ----------
    model : TYPE
        The Keras model.
    solution : TYPE
        A single PyGAD solution as 1D vector.
    data : TYPE
        The data or a generator.
    batch_size : TYPE, optional
        The batch size (i.e. number of samples per step or batch). The default is None. Check documentation of the Keras Model.predict() method for more information.
    verbose : TYPE, optional
        Verbosity mode. The default is 0. Check documentation of the Keras Model.predict() method for more information.
    steps : TYPE, optional
        The total number of steps (batches of samples). The default is None. Check documentation of the Keras Model.predict() method for more information.

    Returns
    -------
    predictions : TYPE
        The Keras model predictions.

    """
    # Fetch the parameters of the best solution.
    solution_weights = model_weights_as_matrix(model=model,
                                               weights_vector=solution)
    _model = tensorflow.keras.models.clone_model(model)
    _model.set_weights(solution_weights)
    predictions = _model.predict(x=data,
                                 batch_size=batch_size,
                                 verbose=verbose,
                                 steps=steps)

    return predictions

class KerasGA:

    def __init__(self, model, num_solutions):

        """
        Creates an instance of the KerasGA class to build a population of model parameters.

        model: A Keras model class.
        num_solutions: Number of solutions in the population. Each solution has different model parameters.
        """
        
        self.model = model

        self.num_solutions = num_solutions

        # A list holding references to all the solutions (i.e. networks) used in the population.
        self.population_weights = self.create_population()

    def create_population(self):

        """
        Creates the initial population of the genetic algorithm as a list of networks' weights (i.e. solutions). Each element in the list holds a different weights of the Keras model.

        The method returns a list holding the weights of all solutions.
        """

        model_weights_vector = model_weights_as_vector(model=self.model)

        net_population_weights = []
        net_population_weights.append(model_weights_vector)

        for idx in range(self.num_solutions-1):

            net_weights = copy.deepcopy(model_weights_vector)
            net_weights = numpy.array(net_weights) + numpy.random.uniform(low=-1.0, high=1.0, size=model_weights_vector.size)

            # Appending the weights to the population.
            net_population_weights.append(net_weights)

        return net_population_weights
