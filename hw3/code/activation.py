###############################################################################
# Imports
###############################################################################
import numpy as np


def sigmoid(inputs):
    '''Calculate sigmoid on an array
    Parameters:
    -----------
        inputs : np.ndarray
            Input to pass through sigmoid
    Returns:
    --------
        np.ndarray
            Output of the sigmoid
    '''
    return 1.0 / (1 + np.exp(-1 * inputs))

def sigmoid_der(inputs):
    '''Calculate derivative of sigmoid
    Parameters:
    -----------
        inputs : np.ndarray
            Input to pass through sigmoid derivative
    Returns:
    --------
        np.ndarray
            Output of the sigmoid derivative
    '''
    sig_input = sigmoid(inputs)
    return sig_input * (1 - sig_input)


if __name__ == '__main__':
    pass

