import pygad.cnn
import numpy

def test_cnn_layers_and_model():
    """Test pygad.cnn layers and Model class."""
    # Dummy data
    data_inputs = numpy.random.uniform(0, 1, (4, 10, 10, 3))
    # The test do not care about the outputs predicted by the network.
    # data_outputs = numpy.array([0, 1, 1, 0])

    input_layer = pygad.cnn.Input2D(input_shape=(10, 10, 3))
    conv_layer = pygad.cnn.Conv2D(num_filters=2,
                                   kernel_size=3,
                                   previous_layer=input_layer,
                                   activation_function="relu")
    max_pooling_layer = pygad.cnn.MaxPooling2D(pool_size=2, 
                                               previous_layer=conv_layer,
                                               stride=2)
    flatten_layer = pygad.cnn.Flatten(previous_layer=max_pooling_layer)
    dense_layer = pygad.cnn.Dense(num_neurons=2, 
                                   previous_layer=flatten_layer,
                                   activation_function="softmax")

    model = pygad.cnn.Model(last_layer=dense_layer,
                            epochs=1,
                            learning_rate=0.01)

    # Test predict
    predictions = model.predict(data_inputs=data_inputs)
    assert len(predictions) == 4

    # Test summary (just to ensure it doesn't crash)
    model.summary()
    
    # Test layers_weights
    weights = pygad.cnn.layers_weights(model)
    assert isinstance(weights, list)
    assert len(weights) > 0

    # Test layers_weights_as_vector
    weights_vector = pygad.cnn.layers_weights_as_vector(model)
    assert isinstance(weights_vector, numpy.ndarray)
    assert weights_vector.ndim == 1
    
    print("test_cnn_layers_and_model passed.")

if __name__ == "__main__":
    test_cnn_layers_and_model()
    print("\nAll CNN tests passed!")
