import copy
import numpy
import tensorflow.keras

def model_weights_as_vector(model):
    """
    Flatten every weight tensor of a Keras model into a single 1D
    NumPy array. Only the weights of trainable layers are included.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The Keras model whose weights should be flattened.

    Returns
    -------
    weights_vector : numpy.ndarray
        A 1D array with every trainable parameter of the model laid
        out in layer order.
    """
    weights_vector = []

    for layer in model.layers: # model.get_weights():
        if layer.trainable:
            layer_weights = layer.get_weights()
            for l_weights in layer_weights:
                vector = numpy.reshape(l_weights, (l_weights.size))
                weights_vector.extend(vector)

    return numpy.array(weights_vector)

def model_weights_as_matrix(model, weights_vector):
    """
    Reshape a flat 1D weights vector back into the per-layer matrices
    expected by ``model.set_weights``. Non-trainable layers keep their
    current weights.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The reference Keras model. Used to read the per-layer shapes.
    weights_vector : array-like
        A 1D vector in the same layout produced by
        ``model_weights_as_vector``.

    Returns
    -------
    weights_matrix : list of numpy.ndarray
        One matrix per layer, ready to be passed to
        ``model.set_weights``.
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
                layer_weights_matrix = numpy.reshape(layer_weights_vector, (layer_weights_shape))
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
    Load the given solution as the model's weights and run a forward
    pass on ``data``. The model is cloned first so the caller's
    instance is left untouched.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The reference Keras model.
    solution : array-like
        A 1D weights vector returned by the GA.
    data : numpy.ndarray or tf.data.Dataset
        Input data passed to ``Model.predict``.
    batch_size : int or None
        Number of samples per step. Forwarded to ``Model.predict``.
    verbose : int
        Verbosity level. Forwarded to ``Model.predict``.
    steps : int or None
        Number of steps (batches). Forwarded to ``Model.predict``.

    Returns
    -------
    predictions : numpy.ndarray
        The Keras model output for ``data``.
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
        Build a population of weight vectors for a Keras model so the
        GA can evolve them.

        Parameters
        ----------
        model : tensorflow.keras.Model
            The Keras model to optimize. Its current weights are used
            as the seed for the first solution.
        num_solutions : int
            Number of solutions in the population. Each solution is a
            flat copy of the model weights with random perturbations
            added to it.
        """

        self.model = model

        self.num_solutions = num_solutions

        # A list holding references to all the solutions (i.e. networks) used in the population.
        self.population_weights = self.create_population()

    def create_population(self):
        """
        Build the initial population. The first solution is the model's
        current flattened weights; every other solution is the same
        vector with a uniform ``[-1, 1]`` perturbation added on top.

        Returns
        -------
        net_population_weights : list of numpy.ndarray
            One flat weight vector per solution.
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
