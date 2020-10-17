###############################################################################
# Imports
###############################################################################
from dataset import Dataset
import numpy as np
from matplotlib import pyplot as plt


class Layer:
    def __init__(
            self,
            num_inputs = 5,
            num_neurons = 3,
    ):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.init_weights()

    def initWeights(self):
        '''Initialize weights for the perceptron
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        '''
        # Uniform initialization from -1 to 1
        self.weights = np.random.uniform(-1, 1, self.num_inputs + 1,
                dtype=np.float64)

    def getWeights(self):
        '''Return weights array
        Parameters:
        -----------
            None
        Returns:
        --------
            np.array of float64
                Array of weight values
        '''
        return self.weights

    def updateWeights(self, updates):
        '''Update the weights of this neuron
        Parameters:
        -----------
            updates : np.ndarray
                updates to apply to the weights
        Returns:
        --------
            None
        '''
        self.weights += updates


if __name__ == '__main__':
    pass

