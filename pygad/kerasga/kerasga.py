import copy
import threading
import weakref
import numpy
import tensorflow.keras

_model_locks = weakref.WeakKeyDictionary()
_model_locks_guard = threading.Lock()


def _model_lock(model):
    """Return the process-local prediction lock for a Keras model.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Model whose temporary weight changes need synchronization.

    Returns
    -------
    lock : threading.RLock
        The existing lock for this model, or a newly registered lock.
        Weak keys let the registry release entries with their models;
        keeping locks outside model state preserves model serialization.
    """
    # Keep locks outside model state so Keras models remain serializable.
    # Weak keys also let models and their locks be released together.
    with _model_locks_guard:
        lock = _model_locks.get(model)
        if lock is None:
            lock = threading.RLock()
            _model_locks[model] = lock
        return lock

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
        One array per weight tensor, ready to be passed to
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
    pass on ``data``. The model's original weights are restored
    afterwards, so the model passed by the caller is not changed.
    Calls sharing the same model are synchronized across threads.

    Parameters
    ----------
    model : tensorflow.keras.Model
        The reference Keras model.
    solution : array-like
        A 1D weights vector returned by the GA.
    data : array-like or input supported by the Keras execution path
        Input passed directly to the model when batch_size and steps are
        both None. Otherwise passed to Model.predict, which also accepts
        inputs such as tf.data.Dataset.
    batch_size : int or None
        Number of samples per step. If set, selects Model.predict rather
        than calling the model directly.
    verbose : int
        Verbosity level forwarded to Model.predict. Not used when calling
        the model directly.
    steps : int or None
        Number of steps (batches). If set, selects Model.predict rather
        than calling the model directly.

    Returns
    -------
    predictions : numpy.ndarray or Keras prediction output
        The direct model output converted to a NumPy array, or the result
        returned by Model.predict when batch_size or steps is specified.

    Notes
    -----
    The model lock covers weight conversion, temporary weight assignment,
    inference, and restoration. Original weights are restored in a finally
    block, including when inference raises an exception. Direct calls to
    model.set_weights outside this helper need their own synchronization.
    """
    with _model_lock(model):
        solution_weights = model_weights_as_matrix(model=model,
                                                   weights_vector=solution)
        original_weights = model.get_weights()
        try:
            model.set_weights(solution_weights)
            if batch_size is None and steps is None:
                predictions = numpy.array(model(data, training=False))
            else:
                predictions = model.predict(x=data,
                                            batch_size=batch_size,
                                            verbose=verbose,
                                            steps=steps)
        finally:
            model.set_weights(original_weights)

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
