###############################################################################
# Imports
###############################################################################
# Custom imports
from settings import *
from autoencoder import Autoencoder
# External imports
import cupy as np
import pathlib
import matplotlib
import matplotlib.pyplot as plt


def drawFeatures(weights, dir_name):
    """Draw the output predictions and save them
    Parameters
    ----------
    weights : np.ndarray
        Array of weights for the neurons
    dir_name : str
        Name of the directory to save the images to
    """
    for i, w in enumerate(weights):
        # Remove bias and normalize on [0, 1]
        w = w[1:]
        w -= w.min()
        w /= (w.max() - w.min())
        w_name = dir_name.joinpath(f'feat_{i:02d}.png')
        matplotlib.image.imsave(str(w_name), w.reshape(28, 28, order='F'),
                cmap='Greys_r')


# Seed for consistency
np.random.seed(SEED)
# Load best weights back up for each model
auto_clean = Autoencoder(input_size=INPUTS)
for weight_file in sorted(AUTO_CLEAN_MODEL_DIR.iterdir()):
    auto_clean.addLayer(file_name=weight_file)
auto_noisy = Autoencoder(input_size=INPUTS)
for weight_file in sorted(AUTO_NOISY_MODEL_DIR.iterdir()):
    auto_noisy.addLayer(file_name=weight_file)
# Neurons to check
neuron_count = auto_clean.layers[0].num_neurons
neurons = np.random.choice(np.arange(neuron_count), 20, replace=False)
clean_weights = auto_clean.layers[0].w[neurons]
drawFeatures(clean_weights, AUTO_CLEAN_IMG_DIR)
noisy_weights = auto_noisy.layers[0].w[neurons]
drawFeatures(noisy_weights, AUTO_NOISY_IMG_DIR)

