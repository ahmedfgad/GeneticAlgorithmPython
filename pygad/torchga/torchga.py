import copy
import numpy
import torch

def model_weights_as_vector(model):
    weights_vector = []

    for curr_weights in model.state_dict().values():
        # Calling detach() to remove the computational graph from the layer.
        # cpu() is called for making shore the data is moved from GPU to cpu
        # numpy() is called for converting the tensor into a NumPy array.
        curr_weights = curr_weights.cpu().detach().numpy()
        vector = numpy.reshape(curr_weights, newshape=(curr_weights.size))
        weights_vector.extend(vector)

    return numpy.array(weights_vector)

def model_weights_as_dict(model, weights_vector):
    weights_dict = model.state_dict()

    start = 0
    for key in weights_dict:
        # Calling detach() to remove the computational graph from the layer. 
        # cpu() is called for making shore the data is moved from GPU to cpu
        # numpy() is called for converting the tensor into a NumPy array.
        w_matrix = weights_dict[key].cpu().detach().numpy()
        layer_weights_shape = w_matrix.shape
        layer_weights_size = w_matrix.size

        layer_weights_vector = weights_vector[start:start + layer_weights_size]
        layer_weights_matrix = numpy.reshape(layer_weights_vector, newshape=(layer_weights_shape))
        weights_dict[key] = torch.from_numpy(layer_weights_matrix)

        start = start + layer_weights_size

    return weights_dict

def predict(model, solution, data):
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
        Creates an instance of the TorchGA class to build a population of model parameters.

        model: A PyTorch model class.
        num_solutions: Number of solutions in the population. Each solution has different model parameters.
        """
        
        self.model = model

        self.num_solutions = num_solutions

        # A list holding references to all the solutions (i.e. networks) used in the population.
        self.population_weights = self.create_population()

    def create_population(self):

        """
        Creates the initial population of the genetic algorithm as a list of networks' weights (i.e. solutions). Each element in the list holds a different weights of the PyTorch model.

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
