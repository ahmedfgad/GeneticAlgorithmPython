import numpy
import pygad.cnn

"""
Convolutional neural network implementation using NumPy
A tutorial that helps to get started (Building Convolutional Neural Network using NumPy from Scratch) available in these links: 
    https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad
    https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a
    https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
It is also translated into Chinese: http://m.aliyun.com/yunqi/articles/585741
"""

train_inputs = numpy.load("../data/dataset_inputs.npy")
train_outputs = numpy.load("../data/dataset_outputs.npy")

sample_shape = train_inputs.shape[1:]
num_classes = 4

input_layer = pygad.cnn.Input2D(input_shape=sample_shape)
conv_layer1 = pygad.cnn.Conv2D(num_filters=2,
                               kernel_size=3,
                               previous_layer=input_layer,
                               activation_function=None)
relu_layer1 = pygad.cnn.Sigmoid(previous_layer=conv_layer1)
average_pooling_layer = pygad.cnn.AveragePooling2D(pool_size=2, 
                                                   previous_layer=relu_layer1,
                                                   stride=2)

conv_layer2 = pygad.cnn.Conv2D(num_filters=3,
                               kernel_size=3,
                               previous_layer=average_pooling_layer,
                               activation_function=None)
relu_layer2 = pygad.cnn.ReLU(previous_layer=conv_layer2)
max_pooling_layer = pygad.cnn.MaxPooling2D(pool_size=2, 
                                           previous_layer=relu_layer2,
                                           stride=2)

conv_layer3 = pygad.cnn.Conv2D(num_filters=1,
                               kernel_size=3,
                               previous_layer=max_pooling_layer,
                               activation_function=None)
relu_layer3 = pygad.cnn.ReLU(previous_layer=conv_layer3)
pooling_layer = pygad.cnn.AveragePooling2D(pool_size=2, 
                                           previous_layer=relu_layer3,
                                           stride=2)

flatten_layer = pygad.cnn.Flatten(previous_layer=pooling_layer)
dense_layer1 = pygad.cnn.Dense(num_neurons=100, 
                               previous_layer=flatten_layer,
                               activation_function="relu")
dense_layer2 = pygad.cnn.Dense(num_neurons=num_classes, 
                               previous_layer=dense_layer1,
                               activation_function="softmax")

model = pygad.cnn.Model(last_layer=dense_layer2,
                        epochs=1,
                        learning_rate=0.01)

model.summary()

model.train(train_inputs=train_inputs, 
            train_outputs=train_outputs)

predictions = model.predict(data_inputs=train_inputs)
print(predictions)

num_wrong = numpy.where(predictions != train_outputs)[0]
num_correct = train_outputs.size - num_wrong.size
accuracy = 100 * (num_correct/train_outputs.size)
print(f"Number of correct classifications : {num_correct}.")
print(f"Number of wrong classifications : {num_wrong.size}.")
print(f"Classification accuracy : {accuracy}.")
