###############################################################################
# Imports
###############################################################################
from activation import *
import numpy as np


class Layer:
    def __init__(
            self,
            num_inputs = 5,
            num_neurons = 3,
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
        self.initWeights()
        # Batch states to save for back-prop
        self.inputs = None # Inputs for the current batch
        self.net_inputs = None # Net inputs pre-activation function
        self.output = None # Final output of the layer
        self.delta = None # Delta of this layer
        self.weight_change = np.zeros(self.weights.shape) # Weight changes for the batch

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
        weights_shape = (self.num_neurons, self.num_inputs + 1)
        self.weights = np.random.uniform(-1, 1, weights_shape)
        self.weight_change = np.zeros(self.weights.shape)

    def forwardPass(self, inputs):
        '''Pass the input forward through the layer
        Parameters:
        -----------
            inputs : np.ndarray of np.float64
                Inputs to the layer with shape (e, i) where e is number of
                elements in the input and i is the number of inputs
        Returns:
        --------
            np.array of np.float64
                Output of the layer with shape (n, i) where n is the number of
                neurons in this layer and i is the number of inputs
        '''
        # Add bias input
        bias = np.ones(inputs.shape[1])
        inputs = np.vstack((bias, inputs))
        # Save inputs for back-prop
        self.inputs = inputs
        # Dot product weights * inputs
        self.net_inputs = np.matmul(self.weights, inputs)
        # Pass through sigmoid activation
        self.output = sigmoid(self.net_inputs)
        return self.output

    def setDelta(self):
        '''Calculate delta value, different for Output and Hidden layer
        Must be implemented per-class
        '''
        raise NotImplementedError('Cannot call from Layer class')

    def updateWeights(self, learning_rate=1.0, alpha=0.1):
        '''Calculate and apply weight updates for the most recent forward pass
        Requires delta to be calculated (varies between hidden/output layers)
        Parameters:
        -----------
            learning_rate : float
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
        self.weight_change *= alpha
        # Calculate new weight change
        new_change = learning_rate * np.matmul(self.delta * self.inputs)
        self.weight_change += new_change
        # Update weights
        self.weights += self.weight_change


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
    layer = Layer(num_inputs=784, num_neurons=10)
    # Create dataset
    dataset = Dataset(DATA_FILE, LABEL_FILE)
    dataset.shuffleData(train=True, test=True)
    # Test batched processing a forward pass
    data, labels = dataset.getTrainBatch(batch_size=5)
    for d, l in zip(data, labels):
        print(layer.forwardPass(d.transpose()).shape)

