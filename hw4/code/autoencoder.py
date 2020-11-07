###############################################################################
# Imports
###############################################################################
from mlp import MLP
import numpy as np
import matplotlib.pyplot as plt


class Autoencoder(MLP):
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
        # Calculate loss
        total_loss = 0
        for d, l in zip(data, labels):
            pred = self.predict(d, one_hot=False)
            loss = np.sum(np.square(l - pred))
            total_loss += loss / 2
        total_loss /= len(data)
        return total_loss


if __name__ == '__main__':
    # Additional imports
    from settings import *
    import pathlib
    import csv
    # Seed for consistency
    np.random.seed(SEED)
    # Delete old model
    for f in AUTO_CLEAN_MODEL_DIR.iterdir():
        f.unlink()
    # Training constants
    # Test network
    autoencoder = Autoencoder(input_size=INPUTS)
    for h in HIDDEN_LAYER_SIZES:
        autoencoder.addLayer(neurons=h, output=False)
    autoencoder.addLayer(neurons=INPUTS, output=True)
    # Train the network
    print('* * * Begin training autoencoder * * *')
    autoencoder.train(
        train_data,
        train_data,
        points_per_epoch=AUTO_CLEAN_POINTS_PER_EPOCH,
        valid_points=AUTO_CLEAN_VALID_POINTS,
        max_epochs=AUTO_CLEAN_MAX_EPOCHS,
        eta=AUTO_CLEAN_ETA,
        alpha=AUTO_CLEAN_ALPHA,
        L=AUTO_CLEAN_L,
        H=AUTO_CLEAN_H,
        patience=AUTO_CLEAN_PATIENCE,
        es_delta=AUTO_CLEAN_ES_DELTA,
        save_dir=AUTO_CLEAN_MODEL_DIR,
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
    plt.ylim([0, max(max(autoencoder.train_err), max(autoencoder.valid_err))])
    plt.savefig(str(AUTO_CLEAN_LOSS_PLOT), bbox_inches='tight', pad_inches=0)
    plt.close()
    # Save parameters to CSV
    print('Saving training parameters')
    csv_rows = list()
    csv_rows.append(['Parameter', 'Value', 'Description'])
    csv_rows.append(['$hidden\\_layer\\_size$', str(HIDDEN_LAYER_SIZES[0]),
        'Neurons in hidden layer'])
    csv_rows.append(['$\\eta$', str(AUTO_CLEAN_ETA), 'Learning rate'])
    csv_rows.append(['$\\alpha$', str(AUTO_CLEAN_ALPHA), 'Momentum'])
    csv_rows.append(['$max\\_epochs$', str(AUTO_CLEAN_MAX_EPOCHS),
        'Maximum training epochs'])
    csv_rows.append(['$L$', str(AUTO_CLEAN_L), 'Lower activation threshold'])
    csv_rows.append(['$H$', str(AUTO_CLEAN_H), 'Upper activation threshold'])
    csv_rows.append(['$patience$', str(AUTO_CLEAN_PATIENCE),
        'Patience before early stopping'])
    csv_rows.append(['$es\\_delta$', str(AUTO_CLEAN_ES_DELTA),
        'Delta value for early stopping'])
    with open(str(AUTO_CLEAN_PARAM_CSV), 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(csv_rows)
    # Write best epoch
    with open(str(AUTO_CLEAN_BEST_EPOCH), 'w') as best_epoch_file:
        best_epoch_file.write(str(autoencoder.best_weights_epoch))

