import copy
import numpy
import torch

def model_weights_as_vector(model):
    """
    Flatten every weight tensor of a PyTorch model into a single 1D
    NumPy array. Tensors are moved off the GPU and detached from the
    computational graph before being converted.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose weights should be flattened.

    Returns
    -------
    weights_vector : numpy.ndarray
        A 1D float array with every parameter of the model laid out
        in ``state_dict`` order.
    """
    weights_vector = []

    for curr_weights in model.state_dict().values():
        # Calling detach() to remove the computational graph from the layer.
        # cpu() is called to make sure the data is moved from the GPU to the CPU.
        # numpy() is called to convert the tensor into a NumPy array.
        curr_weights = curr_weights.cpu().detach().numpy()
        vector = numpy.reshape(curr_weights, (curr_weights.size))
        weights_vector.extend(vector)

    return numpy.array(weights_vector)

def model_weights_as_dict(model, weights_vector):
    """
    Reshape a flat 1D weights vector back into the per-layer tensors
    expected by ``model.load_state_dict``. The shapes are taken from
    the model's current ``state_dict``.

    Parameters
    ----------
    model : torch.nn.Module
        The reference model. Used only to read the per-layer shapes.
    weights_vector : array-like
        A 1D vector in the same layout produced by
        ``model_weights_as_vector``.

    Returns
    -------
    weights_dict : dict
        A dict mapping every parameter name to a freshly built
        ``torch.Tensor`` with the right shape.
    """
    weights_dict = model.state_dict()

    start = 0
    for key in weights_dict:
        # Calling detach() to remove the computational graph from the layer. 
        # cpu() is called to make sure the data is moved from the GPU to the CPU.
        # numpy() is called to convert the tensor into a NumPy array.
        w_matrix = weights_dict[key].cpu().detach().numpy()
        layer_weights_shape = w_matrix.shape
        layer_weights_size = w_matrix.size

        layer_weights_vector = weights_vector[start:start + layer_weights_size]
        layer_weights_matrix = numpy.reshape(layer_weights_vector, (layer_weights_shape))
        weights_dict[key] = torch.from_numpy(layer_weights_matrix)

        start = start + layer_weights_size

    return weights_dict

def predict(model, solution, data):
    """
    Load the given solution as the model's weights and run a forward
    pass on the input ``data``. The model is deep-copied first so the
    caller's instance is left untouched.

    Parameters
    ----------
    model : torch.nn.Module
        The reference model whose architecture should be used for the
        forward pass.
    solution : array-like
        A 1D weights vector returned by the GA.
    data : torch.Tensor
        Input tensor of the right shape for the model.

    Returns
    -------
    predictions : torch.Tensor
        The model's output for ``data``.
    """
    # Fetch the parameters of the best solution.
    model_weights_dict = model_weights_as_dict(model=model,
                                               weights_vector=solution)

    # Use the current solution as the model parameters.
    _model = copy.deepcopy(model)
    _model.load_state_dict(model_weights_dict)

    with torch.no_grad():
        predictions = _model(data)

        return predictions

class TorchGA:

    def __init__(self, model, num_solutions):
        """
        Build a population of weight vectors for a PyTorch model so
        the GA can evolve them.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to optimize. Its current weights are
            used as the seed for the first solution.
        num_solutions : int
            Number of solutions in the population. Each solution is
            a flat copy of the model weights with random perturbations
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
