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
    # Additional imports
    from settings import *
    import pathlib
    import csv
    # Seed for consistency
    np.random.seed(SEED)
    # Delete old model
    for f in CLASS_MODEL_DIR.iterdir():
        f.unlink()
    # Test network
    classifier = Classifier(input_size=INPUTS)
    for h in HIDDEN_LAYER_SIZES:
        classifier.addLayer(neurons=h, output=False)
    classifier.addLayer(neurons=CLASSES, output=True)
    # Train the network
    print('* * * Begin training classifier * * *')
    classifier.train(
        train_data,
        train_labels,
        points_per_epoch=CLASS_POINTS_PER_EPOCH,
        valid_points=CLASS_VALID_POINTS,
        max_epochs=CLASS_MAX_EPOCHS,
        eta=CLASS_ETA,
        alpha=CLASS_ALPHA,
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
    # Save parameters to CSV
    csv_rows = list()
    csv_rows.append(['Parameter', 'Value', 'Description'])
    csv_rows.append(['$hidden\\_layer\\_size$', str(HIDDEN_LAYER_SIZES[0]),
        'Neurons in hidden layer'])
    csv_rows.append(['$\\eta$', str(CLASS_ETA), 'Learning rate'])
    csv_rows.append(['$\\alpha$', str(CLASS_ALPHA), 'Momentum'])
    csv_rows.append(['$max\\_epochs$', str(CLASS_MAX_EPOCHS),
        'Maximum training epochs'])
    csv_rows.append(['$L$', str(CLASS_L), 'Lower activation threshold'])
    csv_rows.append(['$H$', str(CLASS_H), 'Upper activation threshold'])
    csv_rows.append(['$patience$', str(CLASS_PATIENCE),
        'Patience before early stopping'])
    csv_rows.append(['$es\\_delta$', str(CLASS_ES_DELTA),
        'Delta value for early stopping'])
    with open(str(CLASS_PARAM_CSV), 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(csv_rows)
    # Write best epoch
    with open(str(CLASS_BEST_EPOCH), 'w') as best_epoch_file:
        best_epoch_file.write(str(classifier.best_weights_epoch))

