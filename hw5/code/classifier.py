###############################################################################
# Imports
###############################################################################
# Custom imports
from mlp import MLP
# External imports
import numpy as np
import matplotlib.pyplot as plt


class Classifier(MLP):
    def eval(self, data, labels):
        """Evaluate error on a data set
        Parameters
        ----------
        data : np.ndarray
            Array of data points
        labels : np.ndarray
            Array of labels for the data
        Returns
        -------
        float
            Error (1 - Hit Rate)
        """
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
    # Additional imports
    from settings import *
    from sofm import SOFM
    import pathlib
    import csv
    # Seed for consistency
    np.random.seed(SEED)
    # Delete old model
    for f in CLASS_MODEL_DIR.iterdir():
        f.unlink()
    # Load SOFM
    sofm = SOFM(
        SOFM_SHAPE,
        INPUTS,
        weight_file = SOFM_WEIGHT_FILE,
        trainable = False,
    )
    # Map images using SOFM
    train_features = list()
    for x in train_data:
        sofm.forwardPass(x)
        train_features.append(sofm.bmu)
    train_data_map = np.eye(np.prod(SOFM_SHAPE))[train_features]

    # Make MLP
    classifier = Classifier(input_size=np.prod(SOFM_SHAPE))
    classifier.addLayer(neurons=CLASSES, output=True, trainable=True)
    # Train the network
    print('* * * Begin training classifier * * *')
    classifier.train(
        train_data_map,
        train_labels,
        points_per_epoch=CLASS_POINTS_PER_EPOCH,
        valid_points=CLASS_VALID_POINTS,
        max_epochs=CLASS_MAX_EPOCHS,
        eta=CLASS_ETA,
        alpha=CLASS_ALPHA,
        decay=CLASS_DECAY,
        L=CLASS_L,
        H=CLASS_H,
        patience=CLASS_PATIENCE,
        es_delta=CLASS_ES_DELTA,
        save_dir=CLASS_MODEL_DIR,
    )
    # Plot loss over epochs
    plt.figure()
    plt.plot(classifier.epoch_num, classifier.train_err)
    plt.plot(classifier.epoch_num, classifier.valid_err)
    plt.axvline(x=classifier.best_weights_epoch, c='g')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.title(f'Classifier Loss vs Epoch Number')
    plt.xlabel('Epoch number')
    plt.xlim([0, classifier.epoch_num[-1]])
    plt.ylabel('Loss')
    plt.ylim([0, max(max(classifier.train_err), max(classifier.valid_err))])
    plt.savefig(str(CLASS_LOSS_PLOT), bbox_inches='tight', pad_inches=0)
    plt.close()
    # Write best epoch
    with open(str(CLASS_BEST_EPOCH), 'w') as best_epoch_file:
        best_epoch_file.write(str(classifier.best_weights_epoch))

