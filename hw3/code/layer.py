###############################################################################
# Imports
###############################################################################
from activation import *
import numpy as np


class Layer:
    def __init__(
            self,
            num_neurons,
            num_inputs,
        ):
        '''Initialize Layer object
        Parameters:
        -----------
            num_inputs : int
                Number of input elements (not including bias)
            num_neurons : int
                Number of neurons in the layer
        Returns:
        --------
            Layer
                The Layer object which was constructed
        '''
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.initW()
        # Pass states to save for back-prop
        self.x = None # Input for current pass
        self.s = None # Net inputs pre-activation function
        self.y = None # Final output of the layer
        self.delta = None # Delta of this layer
        self.w_change = np.zeros(self.w.shape) # Weight changes

    def initW(self):
        '''Initialize weights for the perceptron
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        '''
        # Uniform initialization from -1 to 1
        w_shape = (self.num_neurons, self.num_inputs + 1)
        self.w = np.random.uniform(-1, 1, w_shape)
        self.w_change = np.zeros(self.w.shape)

    def forwardPass(self, x):
        '''Pass the input forward through the layer
        Parameters:
        -----------
            x : np.ndarray of np.float64
                Input to the layer
        Returns:
        --------
            np.array of np.float64
                Output of the layer
        '''
        # Add bias input
        x = np.concatenate(([1], x))
        # Save inputs for back-prop
        self.x = x
        # Dot product weights * inputs
        self.s = np.matmul(self.w, x)
        # Pass through sigmoid activation
        self.y = sigmoid(self.s)
        return self.y

    def setDelta(self):
        '''Calculate delta value, different for Output and Hidden layer
        Must be implemented per-class
        '''
        raise NotImplementedError('Cannot call from Layer class')

    def getWChange(self, eta=1.0, alpha=0.1):
        '''Calculate weight updates for the most recent forward pass
        Requires delta to be calculated (varies between hidden/output layers)
        Parameters:
        -----------
            eta : float
                Learning rate for weight adjustments
            alpha : float
                Momentum scalar
        Returns:
        --------
            None
        '''
        # Set delta value
        self.setDelta()
        # Pre-scale weight change for momentum
        self.w_change *= alpha
        # Calculate new weight change
        d = self.delta.reshape(len(self.delta), 1)
        x = self.x.reshape(1, len(self.x))
        new_change = eta * np.matmul(d, x)
        self.w_change += new_change

    def changeW(self):
        '''
        '''
        self.w += self.w_change


if __name__ == '__main__':
    # Seed for consistency
    np.random.seed(69420)
    # Additional imports
    from dataset import Dataset
    import pathlib
    # File locations
    CODE_DIR = pathlib.Path(__file__).parent.absolute()
    ROOT_DIR = CODE_DIR.parent
    DATA_FILE = CODE_DIR.joinpath('data.txt')
    LABEL_FILE = CODE_DIR.joinpath('labels.txt')
    IMG_DIR = ROOT_DIR.joinpath('images')
    IMG_DIR.mkdir(mode=0o775, exist_ok=True)
    # Test layer
    layer = Layer(10, 784)
    # Create dataset
    dataset = Dataset(DATA_FILE, LABEL_FILE)
    dataset.shuffleData(train=True, test=True)
    # Test forward pass
    for d, l in zip(dataset.test_data, dataset.test_labels):
        print(layer.forwardPass(d))

