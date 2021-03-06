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
        autoencoder.addLayer(neurons=h, output=False, trainable=True)
    autoencoder.addLayer(neurons=INPUTS, output=True, trainable=True)
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
        decay=AUTO_NOISY_DECAY,
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
    # Write best epoch
    with open(str(AUTO_NOISY_BEST_EPOCH), 'w') as best_epoch_file:
        best_epoch_file.write(str(autoencoder.best_weights_epoch))

