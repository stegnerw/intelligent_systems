###############################################################################
# Imports
###############################################################################
from hidden_layer import HiddenLayer
from output_layer import OutputLayer
import numpy as np


class MLP:
    def __init__(
            self,
            layer_sizes = [784, 100, 10],
            learning_rate = 1,
        ):
        '''Initialize Multi-Layer Perceptron object
        Parameters:
        -----------
            layer_sizes : list of int
                Number of neurons in each layer
                First element is the size of the input data
        Returns:
            MPL
                The MPL object which was constructed
        '''
        # Create layers
        inputs = layer_sizes[:-1]
        neurons = layer_sizes[1:]
        self.layers = list()
        for idx, (i, n) in enumerate(zip(inputs, neurons)):
            # Check output layer
            if (idx + 1) == len(inputs):
                self.layers.append(OutputLayer(num_inputs=i, num_neurons=n))
            else:
                self.layers.append(HiddenLayer(num_inputs=i, num_neurons=n))


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
    mlp = MLP()
    # Create dataset
    dataset = Dataset(DATA_FILE, LABEL_FILE)
    dataset.shuffleData(train=True, test=True)
    data, labels = dataset.getTrainBatch(5)
    # Simulate testing back-prop
    o = mlp.layers[0].forwardPass(data[0].transpose())
    o = mlp.layers[1].forwardPass(o)
    mlp.layers[1].setLabels(labels[0].transpose())
    mlp.layers[1].updateWeights()
