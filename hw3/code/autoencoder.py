###############################################################################
# Imports ###############################################################################
# Custom imports
from settings import *
from dataset import shufflePair
from mlp import MLP
# External imports
import numpy as np
import matplotlib.pyplot as plt


class Autoencoder(MLP):
    def eval(self, data, labels):
        '''Evaluate error on a data set
        Parameters:
        -----------
            data : np.ndarray
                Array of data points
            labels : np.ndarray
                Array of labels for the data
        Returns:
        --------
            float
                Error (1 - Hit Rate)
        '''
        # Draw sample points
        self.drawSamples()
        # Calculate loss
        total_loss = 0
        for d, l in zip(data, labels):
            pred = self.predict(d, one_hot=False)
            loss = np.sum(np.square(l - pred))
            total_loss += loss / 2
        total_loss /= len(data)
        return total_loss

    def addSamplePoints(self, data):
        '''Adds sample points from the given data set
        Parameters:
        -----------
            data : np.ndarray
                Array of data points
        '''
        indeces = np.random.choice(np.arange(len(data)), 8, replace=False)
        self.sample_points = data[indeces]

    def drawSamples(
            self,
            file_name='sample_ae.png',
            title='Autoencoder Sample Outputs'
        ):
        '''Draw the output predictions from self.sample_points
        Parameters:
        -----------
            file_name : str, optional
                Name of the file to save the plot
            title : str, optional
                Title of the plot
        '''
        num_samples = len(self.sample_points)
        plt.figure(figsize=(6.4,2.4))
        plt.suptitle(title)
        for i, d in enumerate(self.sample_points):
            plt.subplot(2, num_samples, i+1)
            plt.imshow(d.reshape(28, 28, order='F'), cmap='Greys_r')
            plt.axis('off')
            pred = self.predict(d, one_hot=False)
            plt.subplot(2, num_samples, i+num_samples+1)
            plt.imshow(pred.reshape(28, 28, order='F'), cmap='Greys_r')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()


if __name__ == '__main__':
    # Seed for consistency
    np.random.seed(69420)
    # Additional imports
    import pathlib
    # File locations
    PLT_NAME = IMG_DIR.joinpath(f'autoencoder_loss.png')
    MODEL_DIR = CODE_DIR.joinpath('autoencoder')
    MODEL_DIR.mkdir(mode=0o775, exist_ok=True)
    # Training constants
    POINTS_PER_EPOCH = 1000
    VALID_POINTS = 1000
    MAX_EPOCHS=1000
    ETA = 0.005
    ALPHA = 0.8
    L = 0
    H = 1
    PATIENCE = 5
    # Define model directory and clear existing weight files
    MODEL_DIR.mkdir(mode=0o775, exist_ok=True)
    for f in MODEL_DIR.iterdir():
        f.unlink()
    # Test network
    autoencoder = Autoencoder(input_size=INPUTS)
    autoencoder.addLayer(neurons=HIDDEN_NEURONS, output=False)
    autoencoder.addLayer(neurons=INPUTS, output=True)
    autoencoder.addSamplePoints(test_data)
    # Train the network
    autoencoder.train(
        train_data,
        train_data,
        points_per_epoch=POINTS_PER_EPOCH,
        valid_points=VALID_POINTS,
        max_epochs=MAX_EPOCHS,
        eta=ETA,
        alpha=ALPHA,
        L=L,
        H=H,
        patience=PATIENCE,
        save_dir=MODEL_DIR,
    )
    plt.figure()
    plt.plot(autoencoder.epoch_num, autoencoder.train_err)
    plt.plot(autoencoder.epoch_num, autoencoder.valid_err)
    plt.axvline(x=autoencoder.best_weights_epoch, c='g')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.title(f'Autoencoder Loss vs Epoch Number')
    plt.xlabel('Epoch number')
    plt.xlim([0, autoencoder.epoch_num[-1]])
    plt.ylabel('Loss')
    plt.ylim([0, max(autoencoder.train_err[0], autoencoder.valid_err[0])])
    plt.savefig(str(PLT_NAME))
    plt.close()

