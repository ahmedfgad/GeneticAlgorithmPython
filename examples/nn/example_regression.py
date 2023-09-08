import numpy
import pygad.nn

"""
This example creates a neural network for regression where the architecture has input and dense layers only. More layers will be added in the future. 
The project only implements the forward pass of a neural network and no training algorithm is used.
For training a neural network using the genetic algorithm, check this project (https://github.com/ahmedfgad/NeuralGenetic) in which the genetic algorithm is used for training the network.
Feel free to leave an issue in this project (https://github.com/ahmedfgad/NumPyANN) in case something is not working properly or to ask for questions. I am also available for e-mails at ahmed.f.gad@gmail.com
"""

# Preparing the NumPy array of the inputs.
data_inputs = numpy.array([[2, 5, -3, 0.1],
                           [8, 15, 20, 13]])

# Preparing the NumPy array of the outputs.
data_outputs = numpy.array([[0.1, 0.2],
                            [1.8, 1.5]])

# The number of inputs (i.e. feature vector length) per sample
num_inputs = data_inputs.shape[1]
# Number of outputs per sample
num_outputs = 1

HL1_neurons = 2

# Building the network architecture.
input_layer = pygad.nn.InputLayer(num_inputs)
hidden_layer1 = pygad.nn.DenseLayer(num_neurons=HL1_neurons, previous_layer=input_layer, activation_function="relu")
output_layer = pygad.nn.DenseLayer(num_neurons=num_outputs, previous_layer=hidden_layer1, activation_function="None")

# Training the network.
pygad.nn.train(num_epochs=100,
               last_layer=output_layer,
               data_inputs=data_inputs,
               data_outputs=data_outputs,
               learning_rate=0.01,
               problem_type="regression")

# Using the trained network for predictions.
predictions = pygad.nn.predict(last_layer=output_layer, 
                         data_inputs=data_inputs, 
                         problem_type="regression")

# Calculating some statistics
abs_error = numpy.mean(numpy.abs(predictions - data_outputs))
print(f"Absolute error : {abs_error}.")
