from ..cnn import cnn
import copy

def population_as_vectors(population_networks):

    """
    Accepts the population as networks and returns a list holding all weights of the CNN layers of each solution (i.e. network) in the population as a vector. 
    If the population has 6 solutions (i.e. networks), this function accepts references to such networks and returns a list with 6 vectors, one for each network (i.e. solution). Each vector holds the weights for all layers for a single CNN.

    population_networks: A list holding references to the CNN models used in the population. 

    Returns a list holding the weights vectors for all solutions (i.e. networks).
    """

    population_vectors = []
    for solution in population_networks:
        # Converting the weights of single layer from the current CNN (i.e. solution) to a vector.
        solution_weights_vector = cnn.layers_weights_as_vector(solution)
        # Appending the weights vector of the current layer of a CNN (i.e. solution) to the weights of the previous layers of the same CNN (i.e. solution).
        population_vectors.append(solution_weights_vector)

    return population_vectors

def population_as_matrices(population_networks, population_vectors):

    """
    Accepts the population as both networks and weights vectors and returns the weights of all layers of each solution (i.e. CNN) in the population as a matrix.
    If the population has 6 solutions (i.e. networks), this function returns a list with 6 matrices, one for each network holding its weights for all layers.

    population_networks: A list holding references to the output (last) layers of the neural networks used in the population. 
    population_vectors: A list holding the weights of all networks as vectors. Such vectors are to be converted into matrices.

    Returns a list holding the weights matrices for all solutions (i.e. networks).
    """

    population_matrices = []
    for solution, solution_weights_vector in zip(population_networks, population_vectors):
        # Converting the weights of single layer from the current CNN (i.e. solution) from a vector to a matrix.
        solution_weights_matrix = cnn.layers_weights_as_matrix(solution, solution_weights_vector)
        # Appending the weights matrix of the current layer of a CNN (i.e. solution) to the weights of the previous layers of the same network (i.e. solution).
        population_matrices.append(solution_weights_matrix)

    return population_matrices

class GACNN:

    def create_population(self):

        """
        Creates the initial population of the genetic algorithm as a list of CNNs (i.e. solutions). Each element in the list holds a reference to the instance of the cnn.Model class. 

        The method returns the list holding the references to the CNN models.
        """

        population_networks = []
        for solution in range(self.num_solutions):

            network = copy.deepcopy(self.model)

            # Appending the CNN model to the list of population networks.
            population_networks.append(network)

        return population_networks

    def __init__(self, model, num_solutions):

        """
        Creates an instance of the GACNN class for training a CNN using the genetic algorithm.
        The constructor of the GACNN class creates an initial population of multiple CNNs using the create_population() method.
        The population returned holds references to instances of the cnn.Model class.

        model: An instance of the pygad.cnn.Model class representing the architecture of all solutions in the population.
        num_solutions: Number of CNNs (i.e. solutions) in the population. Based on the value passed to this parameter, a number of identical CNNs are created where their parameters are optimized using the genetic algorithm.
        """
        
        self.model = model

        self.num_solutions = num_solutions

        # A list holding references to all the solutions (i.e. CNNs) used in the population.
        self.population_networks = self.create_population()

    def update_population_trained_weights(self, population_trained_weights):

        """
        The `update_population_trained_weights()` method updates the `trained_weights` attribute of each CNN according to the weights passed in the `population_trained_weights` parameter.

        population_trained_weights: A list holding the trained weights of all networks as matrices. Such matrices are to be assigned to the 'trained_weights' attribute of all layers of all CNNs.
        """

        idx = 0
        # Fetches all layers weights matrices for a single solution (i.e. CNN)
        for solution in self.population_networks:
            # Calling the cnn.update_layers_trained_weights() function for updating the 'trained_weights' attribute for all layers in the current solution (i.e. CNN).
            cnn.update_layers_trained_weights(model=solution, 
                                              final_weights=population_trained_weights[idx])
            idx = idx + 1
