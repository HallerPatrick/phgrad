import numpy as np


def he_initialization(shape):
    """
    Initialize a weight matrix with the He normal initialization.

    Parameters:
    shape (tuple): The shape of the weight matrix. Typically (number of units in the previous layer, number of units in the current layer).

    Returns:
    numpy.ndarray: An array of weights initialized according to the He normal distribution.
    """
    fan_in = shape[0]  # Number of input units
    std_dev = np.sqrt(2 / fan_in)  # Standard deviation for He initialization
    return np.random.normal(0, std_dev, shape)
