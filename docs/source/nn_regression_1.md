# Regression Example 1

The next code listing builds a neural network for regression. Here is what to do to make the code works for regression:

1. Set the `problem_type` parameter in the `pygad.nn.train()` and `pygad.nn.predict()` functions to the string `"regression"`.

```python
pygad.nn.train(...,
               problem_type="regression")

predictions = pygad.nn.predict(..., 
                               problem_type="regression")
```

2. Set the activation function for the output layer to the string `"None"`. 

```python
output_layer = pygad.nn.DenseLayer(num_neurons=num_outputs, previous_layer=hidden_layer1, activation_function="None")
```

3. Calculate the prediction error according to your preferred error function. Here is how the mean absolute error is calculated.

```python
abs_error = numpy.mean(numpy.abs(predictions - data_outputs))
print(f"Absolute error : {abs_error}.")
```

Here is the complete code. Yet, there is no algorithm used to train the network and thus the network is expected to give bad results. Later, the `pygad.gann` module is used to train either a regression or classification networks. 

```python
import numpy
import pygad.nn

# Preparing the NumPy array of the inputs.
data_inputs = numpy.array([[2, 5, -3, 0.1],
                           [8, 15, 20, 13]])

# Preparing the NumPy array of the outputs.
data_outputs = numpy.array([0.1, 
                            1.5])

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
```
