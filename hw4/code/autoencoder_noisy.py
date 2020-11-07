###############################################################################
# Imports
###############################################################################
from autoencoder import Autoencoder
from settings import *
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import csv


if __name__ == '__main__':
    # Seed for consistency
    np.random.seed(SEED)
    # Delete old model
    for f in AUTO_NOISY_MODEL_DIR.iterdir():
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
        noisy_train_data,
        train_data,
        points_per_epoch=AUTO_NOISY_POINTS_PER_EPOCH,
        valid_points=AUTO_NOISY_VALID_POINTS,
        max_epochs=AUTO_NOISY_MAX_EPOCHS,
        eta=AUTO_NOISY_ETA,
        alpha=AUTO_NOISY_ALPHA,
        L=AUTO_NOISY_L,
        H=AUTO_NOISY_H,
        patience=AUTO_NOISY_PATIENCE,
        es_delta=AUTO_NOISY_ES_DELTA,
        save_dir=AUTO_NOISY_MODEL_DIR,
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
    plt.savefig(str(AUTO_NOISY_LOSS_PLOT), bbox_inches='tight', pad_inches=0)
    plt.close()
    # Save parameters to CSV
    print('Saving training parameters')
    csv_rows = list()
    csv_rows.append(['Parameter', 'Value', 'Description'])
    csv_rows.append(['$hidden\\_layer\\_size$', str(HIDDEN_LAYER_SIZES[0]),
        'Neurons in hidden layer'])
    csv_rows.append(['$\\eta$', str(AUTO_NOISY_ETA), 'Learning rate'])
    csv_rows.append(['$\\alpha$', str(AUTO_NOISY_ALPHA), 'Momentum'])
    csv_rows.append(['$max\\_epochs$', str(AUTO_NOISY_MAX_EPOCHS),
        'Maximum training epochs'])
    csv_rows.append(['$L$', str(AUTO_NOISY_L), 'Lower activation threshold'])
    csv_rows.append(['$H$', str(AUTO_NOISY_H), 'Upper activation threshold'])
    csv_rows.append(['$patience$', str(AUTO_NOISY_PATIENCE),
        'Patience before early stopping'])
    csv_rows.append(['$es\\_delta$', str(AUTO_NOISY_ES_DELTA),
        'Delta value for early stopping'])
    with open(str(AUTO_NOISY_PARAM_CSV), 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(csv_rows)
    # Write best epoch
    with open(str(AUTO_NOISY_BEST_EPOCH), 'w') as best_epoch_file:
        best_epoch_file.write(str(autoencoder.best_weights_epoch))

