# Image Classification

This example is discussed in the **Steps to Build a Neural Network** section and its complete code is listed below. 

Remember to either download or create the [dataset_features.npy](https://github.com/ahmedfgad/NumPyANN/blob/master/dataset_features.npy) and [outputs.npy](https://github.com/ahmedfgad/NumPyANN/blob/master/outputs.npy) files before running this code.

```python
import numpy
import pygad.nn

# Reading the data features. Check the 'extract_features.py' script for extracting the features & preparing the outputs of the dataset.
data_inputs = numpy.load("dataset_features.npy") # Download from https://github.com/ahmedfgad/NumPyANN/blob/master/dataset_features.npy

# Optional step for filtering the features using the standard deviation.
features_STDs = numpy.std(a=data_inputs, axis=0)
data_inputs = data_inputs[:, features_STDs > 50]

# Reading the data outputs. Check the 'extract_features.py' script for extracting the features & preparing the outputs of the dataset.
data_outputs = numpy.load("outputs.npy") # Download from https://github.com/ahmedfgad/NumPyANN/blob/master/outputs.npy

# The number of inputs (i.e. feature vector length) per sample
num_inputs = data_inputs.shape[1]
# Number of outputs per sample
num_outputs = 4

HL1_neurons = 150
HL2_neurons = 60

# Building the network architecture.
input_layer = pygad.nn.InputLayer(num_inputs)
hidden_layer1 = pygad.nn.DenseLayer(num_neurons=HL1_neurons, previous_layer=input_layer, activation_function="relu")
hidden_layer2 = pygad.nn.DenseLayer(num_neurons=HL2_neurons, previous_layer=hidden_layer1, activation_function="relu")
output_layer = pygad.nn.DenseLayer(num_neurons=num_outputs, previous_layer=hidden_layer2, activation_function="softmax")

# Training the network.
pygad.nn.train(num_epochs=10,
               last_layer=output_layer,
               data_inputs=data_inputs,
               data_outputs=data_outputs,
               learning_rate=0.01)

# Using the trained network for predictions.
predictions = pygad.nn.predict(last_layer=output_layer, data_inputs=data_inputs)

# Calculating some statistics
num_wrong = numpy.where(predictions != data_outputs)[0]
num_correct = data_outputs.size - num_wrong.size
accuracy = 100 * (num_correct/data_outputs.size)
print(f"Number of correct classifications : {num_correct}.")
print(f"Number of wrong classifications : {num_wrong.size}.")
print(f"Classification accuracy : {accuracy}.")
```

<!-- python-examples
nn/example_classification.py
nn/extract_features.py
-->

**Python examples**

**[Image classification](../../examples/nn/example_classification.py)**

Classify fruit images from prepared feature vectors.

`examples/nn/example_classification.py`

<details>
<summary>Run this example</summary>

**Requires:** PyGAD

**Data:** examples/data/dataset_features.npy and examples/data/outputs.npy. See the [dataset setup instructions](../../examples/data/README.md).

From the repository root, change to examples/nn/ so the relative data paths resolve:

```console
cd examples/nn
python example_classification.py
```

</details>

**[Prepare image features](../../examples/nn/extract_features.py)**

Extract fruit-image features and write the arrays used by the dense classifiers.

`examples/nn/extract_features.py`

<details>
<summary>Run this example</summary>

**Requires:** PyGAD, scikit-image

**Data:** The apple, lemon, mango, and raspberry folders under examples/data/Fruit360/. See the [dataset setup instructions](../../examples/data/README.md).

From the repository root, change to examples/nn/ so the relative data paths resolve:

```console
cd examples/nn
python extract_features.py
```

</details>

<!-- /python-examples -->
