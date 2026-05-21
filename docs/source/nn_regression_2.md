# Regression Example 2 - Fish Weight Prediction

This example uses the Fish Market Dataset available at Kaggle (https://www.kaggle.com/aungpyaeap/fish-market). Simply download the CSV dataset from [this link](https://www.kaggle.com/aungpyaeap/fish-market/download) (https://www.kaggle.com/aungpyaeap/fish-market/download). The dataset is also available at the [GitHub project of the pygad.nn module](https://github.com/ahmedfgad/NumPyANN): https://github.com/ahmedfgad/NumPyANN

Using the Pandas library, the dataset is read using the `read_csv()` function. 

```python
data = numpy.array(pandas.read_csv("Fish.csv"))
```

The last 5 columns in the dataset are used as inputs and the **Weight** column is used as output.

```python
# Preparing the NumPy array of the inputs.
data_inputs = numpy.asarray(data[:, 2:], dtype=numpy.float32)

# Preparing the NumPy array of the outputs.
data_outputs = numpy.asarray(data[:, 1], dtype=numpy.float32) # Fish Weight
```

Note how the activation function at the last layer is set to `"None"`. Moreover, the `problem_type` parameter in the `pygad.nn.train()` and `pygad.nn.predict()` functions is set to `"regression"`.

After the `pygad.nn.train()` function completes, the mean absolute error is calculated.

```python
abs_error = numpy.mean(numpy.abs(predictions - data_outputs))
print(f"Absolute error : {abs_error}.")
```

Here is the complete code. 

```python
import numpy
import pygad.nn
import pandas

data = numpy.array(pandas.read_csv("Fish.csv"))

# Preparing the NumPy array of the inputs.
data_inputs = numpy.asarray(data[:, 2:], dtype=numpy.float32)

# Preparing the NumPy array of the outputs.
data_outputs = numpy.asarray(data[:, 1], dtype=numpy.float32) # Fish Weight

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
