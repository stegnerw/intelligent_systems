###############################################################################
# Imports
###############################################################################
# Custom imports
from settings import *
from classifier import Classifier
from autoencoder import Autoencoder
# External imports
import numpy as np
import pathlib
import matplotlib
import matplotlib.pyplot as plt


def drawFeatures(weights, dir_name):
    '''Draw the output predictions and save them
    Parameters:
    -----------
        weights : np.ndarray
            Array of weights for the neurons
        dir_name : str
            Name of the directory to save the images to
    '''
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
autoencoder = Autoencoder(input_size=INPUTS)
for weight_file in sorted(AUTO_MODEL_DIR.iterdir()):
    autoencoder.addLayer(file_name=weight_file)
classifier = Classifier(input_size=INPUTS)
for weight_file in sorted(CLASS_MODEL_DIR.iterdir()):
    classifier.addLayer(file_name=weight_file)
# Neurons to check
neuron_count = classifier.layers[0].num_neurons
neurons = np.random.choice(np.arange(neuron_count), 20, replace=False)
class_weights = classifier.layers[0].w[neurons]
drawFeatures(class_weights, CLASS_FEAT_DIR)
auto_weights = autoencoder.layers[0].w[neurons]
drawFeatures(auto_weights, AUTO_FEAT_DIR)

