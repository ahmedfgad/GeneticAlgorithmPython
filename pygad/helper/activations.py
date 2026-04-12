import numpy

def sigmoid(sop):
    """
    Applies the sigmoid function.

    sop: The input to which the sigmoid function is applied.

    Returns the result of the sigmoid function.
    """

    if type(sop) in [list, tuple]:
        sop = numpy.array(sop)

    return 1.0 / (1 + numpy.exp(-1 * sop))

def relu(sop):
    """
    Applies the ReLU function.

    sop: The input to which the relu function is applied.

    Returns the result of the ReLU function.
    """

    if not (type(sop) in [list, tuple, numpy.ndarray]):
        if sop < 0:
            return 0
        else:
            return sop
    elif type(sop) in [list, tuple]:
        sop = numpy.array(sop)

    result = sop
    result[sop < 0] = 0

    return result

def softmax(layer_outputs):
    """
    Applies the softmax function.

    sop: The input to which the softmax function is applied.

    Returns the result of the softmax function.
    """
    return layer_outputs / (numpy.sum(layer_outputs) + 0.000001)
