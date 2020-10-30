###############################################################################
# Imports
###############################################################################
# Custom imports
from settings import *
from dataset import shufflePair
from mlp import MLP
# External imports
import numpy as np
import matplotlib.pyplot as plt


class Classifier(MLP):
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
        correct = 0
        for d, l in zip(data, labels):
            # Make prediction
            pred = self.predict(d, one_hot=True)
            # Check equality with label
            if (np.sum(np.abs(pred - l)) == 0):
                correct += 1
        hit_rate = correct / len(data)
        return (1 - hit_rate)


if __name__ == '__main__':
    # Seed for consistency
    np.random.seed(69420)
    # Additional imports
    import pathlib
    # File locations
    PLT_NAME = IMG_DIR.joinpath(f'classifier_loss.png')
    MODEL_DIR = CODE_DIR.joinpath('classifier')
    MODEL_DIR.mkdir(mode=0o775, exist_ok=True)
    for f in MODEL_DIR.iterdir():
        f.unlink()
    # Training constants
    POINTS_PER_EPOCH = 1000
    VALID_POINTS = 1000
    MAX_EPOCHS=1000
    ETA = 0.01
    ALPHA = 0.8
    L = 0.25
    H = 0.75
    PATIENCE = 5
    # Test network
    classifier = Classifier(input_size=INPUTS)
    classifier.addLayer(neurons=HIDDEN_NEURONS, output=False)
    classifier.addLayer(neurons=CLASSES, output=True)
   # Train the network
    classifier.train(
        train_data,
        train_labels,
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
    plt.plot(classifier.epoch_num, classifier.train_err)
    plt.plot(classifier.epoch_num, classifier.valid_err)
    plt.axvline(x=classifier.best_weights_epoch, c='g')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.title(f'Classifier Loss vs Epoch Number')
    plt.xlabel('Epoch number')
    plt.xlim([0, classifier.epoch_num[-1]])
    plt.ylabel('Loss')
    plt.ylim([0, max(classifier.train_err[0], classifier.valid_err[0])])
    plt.savefig(str(PLT_NAME))
    plt.close()

