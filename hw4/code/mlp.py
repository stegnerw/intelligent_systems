###############################################################################
# Imports
###############################################################################
# Custom imports
from settings import *
from dataset import shufflePair
from hidden_layer import HiddenLayer
from output_layer import OutputLayer
# External imports
import numpy as np
import enlighten # Progress bar for training


class MLP:
    def __init__(
            self,
            input_size=INPUTS,
        ):
        """Initialize Multi-Layer Perceptron object
        Parameters
        ----------
        input_size : int
            Number of inputs to the network
        """
        # Initialize parameters
        self.layers = list()
        self.input_size = input_size

    def addLayer(self, file_name=None, neurons=None, output=False,
            trainable=True):
        """Add a layer to the network
        """
        if len(self.layers) == 0:
            input_size = self.input_size
        else:
            input_size = self.layers[-1].num_neurons
        if file_name:
            if output:
                self.layers.append(OutputLayer(weight_file=file_name,
                    trainable=trainable))
            else:
                self.layers.append(HiddenLayer(weight_file=file_name,
                    trainable=trainable))
        else:
            if output:
                self.layers.append(OutputLayer(num_neurons=neurons,
                    inputs=input_size, trainable=trainable))
            else:
                self.layers.append(HiddenLayer(num_neurons=neurons,
                    inputs=input_size, trainable=trainable))

    def predict(self, data, one_hot):
        """Predict the output given an input
        Parameters
        ----------
        data : np.ndarray
            Data point to predict on
        one_hot : bool
            Whether the output should be one-hot or raw
        Returns
        -------
        np.ndarray
            Array of prediction
        """
        # Forward pass through all the layers
        layer_output = data
        for l in self.layers:
            layer_output = l.forwardPass(layer_output)
        if one_hot:
            max_idx = np.argmax(layer_output)
            pred = np.eye(10)[max_idx]
        else:
            pred = layer_output
        return pred

    def trainPoint(self, data, label, eta, alpha, decay, L, H):
        """Update the weights for a single point
        Parameters
        ----------
        data : np.ndarray
            The data point
        labels : np.ndarray
            One-hot encoded label
        eta : float
            Learning rate
        alpha : float
            Momentum scalar
        decay : float
            Weight decay scalar
        L, H : float, optional
            Low and high thresholds for training
        """
        # Weight change for last layer
        self.predict(data, one_hot=False)
        self.layers[-1].setLabel(label)
        self.layers[-1].thresholdOutputs(L, H)
        self.layers[-1].getWChange(eta, alpha, decay)
        # Back-prop error
        for i in range(len(self.layers)-2, -1, -1):
            # Stop the back-prop if we hit an untrainable layer
            if self.layers[i].trainable == False:
                break
            down_w = self.layers[i+1].w
            down_delta = self.layers[i+1].delta
            self.layers[i].setDownstreamSum(down_w, down_delta)
            self.layers[i].getWChange(eta, alpha, decay)
        # Apply weight changes
        for l in self.layers:
            l.changeW()

    def logError(self,
            train_data,
            train_labels,
            valid_data,
            valid_labels,
            epoch_num,
        ):
        """Log error for train/test data for a given epoch
        Parameters
        ----------
        train_data : np.ndarray
            Array of data points for the train set
        train_labels : np.ndarray
            Array of labels for the train set
        valid_data : np.ndarray
            Array of data points for the validation set
        valid_labels : np.ndarray
            Array of labels for the validation set
        epoch_num : int
            The current epoch number
        """
        # Evaluate and log train error
        train_err = self.eval(train_data, train_labels)
        self.train_err.append(train_err)
        # Evaluate and log test error
        valid_err = self.eval(valid_data, valid_labels)
        self.valid_err.append(valid_err)
        # Record epoch number
        self.epoch_num.append(epoch_num)
        # Print out metrics
        print(f'Epoch {epoch_num}')
        print(f'\tTrain Loss:\t\t{train_err:0.03f}')
        print(f'\tValidation Loss:\t{valid_err:0.03f}')

    def train(self,
            data,
            labels,
            points_per_epoch,
            valid_points,
            max_epochs,
            eta,
            alpha,
            decay,
            L,
            H,
            patience,
            es_delta,
            save_dir,
        ):
        """Train the network up to the desired number of epochs
        Parameters:
        -----------
        data : np.ndarray
            Array of training data points
        labels : np.ndarray
            Array of training labels
        points_per_epoch : int
            Number of training points to use in each epoch
        valid_points : int
            Number of training points to set aside for validation
            These points are set aside before training
        max_epochs : int
            Maximum number of epochs to train
        eta, alpha, decay : float
            Learning rate, Momentum scalar, and Weight decay scalar
        L, H : float
            Low and high thresholds for training
        patience : int
            Amount of epochs with no improvement before early stopping
        es_delta : float
            Required improvement for early stopping
        save_dir : pathlib.Path or str
            Directory to save best model parameters in
        """
        # Initialize progress bar
        pbar_manager = enlighten.get_manager()
        pbar = pbar_manager.counter(total=max_epochs, desc='Training',
                unit = 'epochs', leave=False)
        # Lists to track metrics
        self.train_err = list()
        self.valid_err = list()
        self.epoch_num = list()
        # Set aside validation partition (after shuffling)
        assert valid_points + points_per_epoch <= len(data), \
                'Not enough data for validation points and points per epoch'
        shufflePair(data, labels)
        valid_data = data[:valid_points]
        valid_labels = labels[:valid_points]
        train_data = data[valid_points:]
        train_labels = labels[valid_points:]
        # Log initial accuracy (using whole train partition)
        self.logError(
                train_data,
                train_labels,
                valid_data,
                valid_labels,
                0,
        )
        # Iterate through epochs or until early stopping
        impatience = 0
        self.best_weights_epoch = 0
        best_loss = np.inf
        for e in range(1, max_epochs+1):
            pbar.update()
            shufflePair(train_data, train_labels)
            epoch_train_data = train_data[:points_per_epoch]
            epoch_train_labels = train_labels[:points_per_epoch]
            for d, l in zip(epoch_train_data, epoch_train_labels):
                self.trainPoint(d, l, eta, alpha, decay, L, H)
            # Log data every 10 epochs
            if (e % 10) == 0:
                self.logError(
                        epoch_train_data,
                        epoch_train_labels,
                        valid_data,
                        valid_labels,
                        e,
                )
                # Check for early stopping
                if ((best_loss - self.valid_err[-1]) >= es_delta):
                    impatience = 0
                    self.best_weights_epoch = e
                    best_loss = self.valid_err[-1]
                    for i, l in enumerate(self.layers):
                        layer_name = save_dir.joinpath(f'layer_{i:02d}')
                        l.saveWeights(layer_name)
                else:
                    impatience += 1
                    # We have become too impatient
                    if impatience >= patience:
                        print(f'* * * Early stopping hit * * *')
                        break
        pbar.close()


if __name__ == '__main__':
    print('Warning: This file does not do anything.')
    print('Run either classifier.py or autoencoder.py.')

