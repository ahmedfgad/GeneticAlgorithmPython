import numpy
import pygad
import pygad.kerasga
import tensorflow.keras
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import pytest


@pytest.mark.parametrize("batched", [False, True])
def test_shared_model_predictions_preserve_solution_and_original_weights(batched):
    inputs = tensorflow.keras.layers.Input(shape=(1,))
    outputs = tensorflow.keras.layers.Dense(1, use_bias=False)(inputs)
    model = tensorflow.keras.Model(inputs=inputs, outputs=outputs)
    model.set_weights([numpy.array([[9.]], dtype=numpy.float32)])
    original_call = model.call
    state_lock = threading.Lock()
    state = {"active": 0, "maximum": 0}
    def observed_call(*args, **kwargs):
        with state_lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        try:
            # Allow competing predictions to enter while this one is active.
            time.sleep(0.01)
            return original_call(*args, **kwargs)
        finally:
            with state_lock:
                state["active"] -= 1
    model.call = observed_call
    # Run eagerly so the observation surrounds every prediction, not tracing.
    model.compile(run_eagerly=True)
    start = threading.Barrier(2)
    def predict(value):
        start.wait(timeout=10)
        return pygad.kerasga.predict(model, numpy.array([value]),
                                     numpy.array([[1.], [2.]], dtype=numpy.float32),
                                     batch_size=2 if batched else None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(predict, 1.)
        second = pool.submit(predict, 2.)
        numpy.testing.assert_allclose(first.result(timeout=30), [[1.], [2.]])
        numpy.testing.assert_allclose(second.result(timeout=30), [[2.], [4.]])
    assert state["maximum"] == 1
    numpy.testing.assert_array_equal(model.get_weights()[0], [[9.]])


def test_predict_restores_weights_after_failure():
    inputs = tensorflow.keras.layers.Input(shape=(1,))
    model = tensorflow.keras.Model(inputs=inputs,
        outputs=tensorflow.keras.layers.Dense(1, use_bias=False)(inputs))
    model.set_weights([numpy.array([[9.]], dtype=numpy.float32)])
    def fail(*args, **kwargs):
        raise ValueError("prediction failed")
    model.call = fail
    with pytest.raises(ValueError, match="prediction failed"):
        pygad.kerasga.predict(model, numpy.array([1.]), numpy.array([[1.]]))
    numpy.testing.assert_array_equal(model.get_weights()[0], [[9.]])

def test_kerasga_evolution():
    """Test pygad.kerasga with pygad.GA."""

    # XOR data
    data_inputs = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    data_outputs = numpy.array([[1, 0], [0, 1], [0, 1], [1, 0]]) # One-hot encoded

    input_layer = tensorflow.keras.layers.Input(shape=(2,))
    dense_layer = tensorflow.keras.layers.Dense(4, activation="relu")(input_layer)
    output_layer = tensorflow.keras.layers.Dense(2, activation="softmax")(dense_layer)

    model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)

    keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=10)

    def fitness_func(ga_instance, solution, solution_idx):
        model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                     weights_vector=solution)
        model.set_weights(weights=model_weights_matrix)
        predictions = model.predict(data_inputs, verbose=0)
        
        cce = tensorflow.keras.losses.CategoricalCrossentropy()
        loss = cce(data_outputs, predictions).numpy()
        fitness = 1.0 / (loss + 0.00000001)
        return fitness

    ga_instance = pygad.GA(num_generations=2,
                           num_parents_mating=5,
                           initial_population=keras_ga.population_weights,
                           fitness_func=fitness_func,
                           suppress_warnings=True)

    ga_instance.run()
    assert ga_instance.run_completed
    assert ga_instance.generations_completed == 2
    
    print("test_kerasga_evolution passed.")

if __name__ == "__main__":
    test_kerasga_evolution()
    print("\nAll KerasGA tests passed!")
