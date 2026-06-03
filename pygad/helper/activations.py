import numpy

def sigmoid(sop):
    """
    Apply the sigmoid activation function element-wise:
    ``sigmoid(x) = 1 / (1 + exp(-x))``.

    Parameters
    ----------
    sop : numeric, list, tuple, or numpy.ndarray
        The input value(s). Lists and tuples are converted to a
        numpy array before computing.

    Returns
    -------
    activated : numeric or numpy.ndarray
        The element-wise sigmoid of the input.
    """

    if type(sop) in [list, tuple]:
        sop = numpy.array(sop)

    return 1.0 / (1 + numpy.exp(-1 * sop))

def relu(sop):
    """
    Apply the ReLU activation function element-wise:
    ``relu(x) = max(0, x)``.

    Parameters
    ----------
    sop : numeric, list, tuple, or numpy.ndarray
        The input value(s). Scalars are handled as a special case.
        Lists and tuples are converted to a numpy array.

    Returns
    -------
    activated : numeric or numpy.ndarray
        The element-wise ReLU of the input.
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
    Apply a sum-normalized softmax: divide each value by the sum of
    all values plus a tiny constant to avoid division by zero.

    Note that this is not the canonical softmax (which uses
    exponentials); it just normalizes the inputs so they sum to one.

    Parameters
    ----------
    layer_outputs : numpy.ndarray
        The values to normalize.

    Returns
    -------
    activated : numpy.ndarray
        The normalized values.
    """
    return layer_outputs / (numpy.sum(layer_outputs) + 0.000001)
