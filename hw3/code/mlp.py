###############################################################################
# Imports
###############################################################################
from hidden_layer import HiddenLayer
from output_layer import OutputLayer
import numpy as np
import enlighten # Progress bar for training


class MLP:
    def __init__(
            self,
            layer_sizes = [784, 128, 10],
            eta = 0.001,
            alpha = 0.9,
            H = 0.75,
            L = 0.25,
        ):
        '''Initialize Multi-Layer Perceptron object
        Parameters:
        -----------
            layer_sizes : list of int
                Number of neurons in each layer
                First element is the size of the input data
            eta : float
                Learning rate
            alpha : float
                Momentum scalar
            H, L : float, optional
                High and low thresholds for training
        Returns:
            MPL
                The MPL object which was constructed
        '''
        self.layers = list()
        self.eta = eta
        self.alpha = alpha
        self.H = H
        self.L = L
        # Create layers
        neurons = layer_sizes[1:]
        inputs = layer_sizes[:-1]
        for idx, (n, i) in enumerate(zip(neurons, inputs)):
            # Check output layer
            if (idx + 1) == len(inputs):
                self.layers.append(OutputLayer(n, i))
            else:
                self.layers.append(HiddenLayer(n, i))

    def trainBatch(self, data, labels):
        '''Update the weights for a single batch
        For now, a batch is just 1 data point
        Parameters:
        -----------
            data : np.ndarray
                Array of data for the batch
            labels : np.ndarray
                Array of labels for the batch
        Returns:
        --------
            None
        '''
        # Forward pass through all the layers
        layer_output = data
        for l in self.layers:
            layer_output = l.forwardPass(layer_output)
        # Threshold output layer
        self.layers[-1].thresholdOutputs(self.L, self.H)
        # Weight change for last layer
        self.layers[-1].setLabel(labels)
        self.layers[-1].getWChange()
        # Back-prop error
        for i in range(len(mlp.layers)-2, -1, -1):
            down_w = self.layers[i+1].w
            down_delta = self.layers[i+1].delta
            self.layers[i].setDownstreamSum(down_w, down_delta)
            self.layers[i].getWChange(self.eta, self.alpha)
        # Apply weight changes
        for l in self.layers:
            l.changeW()

    def predict(self, data):
        '''Predict and threshold output of
        Parameters:
        -----------
            data : np.ndarray
                Data point to predict on
        Returns:
        --------
            np.ndarray
                One-hot array of prediction
        '''
        # Forward pass through all the layers
        layer_output = data
        for l in self.layers:
            layer_output = l.forwardPass(layer_output)
        # Threshold final output
        max_idx = np.argmax(layer_output)
        return np.eye(10)[max_idx]

    def eval(self, test_data, test_labels):
        '''Evaluate accuracy on a data set
        For now, a batch is just 1 data point
        Parameters:
        -----------
            test_data : np.ndarray
                Array of data points
            test_labels : np.ndarray
                Array of labels for the data
        Returns:
        --------
            float
                Balanced accuracy
        '''
        correct = 0
        for data, label in zip(test_data, test_labels):
            # Make prediction
            pred = self.predict(data)
            # Check equality with label
            if (np.sum(np.abs(pred - label)) == 0):
                correct += 1
        return correct / len(test_data)

    def train(self, dataset, epochs):
        '''Train the network for the desired number of epochs
        Parameters:
        -----------
            dataset : Dataset
                Prepped dataset object for training
            epochs : int
                Number of epochs to train
        Returns:
        --------
            None
        '''
        # Initialize progress bar
        pbar = enlighten.Counter(total = epochs, desc = 'Training',
                unit = 'epochs')
        # Lists to track metrics
        self.bal_acc = list()
        self.epoch_num = list()
        # Iterate through epochs
        for e in range(epochs):
            # Shuffle data
            dataset.shuffleData()
            # Log data every 10 epochs
            if (e % 10) == 0:
                self.bal_acc.append(self.eval(dataset.train_data,
                    dataset.train_labels))
                self.epoch_num.append(e)
                print(f'Epoch: {e}\tBalanced Accuracy: {self.bal_acc[-1]}')
            for data, labels in zip(dataset.train_data[:1000], dataset.train_labels[:1000]):
                self.trainBatch(data, labels)
            # Update progress bar
            pbar.update()


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
    # Create dataset
    dataset = Dataset(DATA_FILE, LABEL_FILE)
    # Test layer
    mlp = MLP(
        layer_sizes = [784, 100, 10],
        eta = 0.1,
        alpha = 0.9,
        H = 0.75,
        L = 0.25,
    )
    # Train for 100 epochs
    mlp.train(dataset, 200)

