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
        w = w[1:]
        w_name = dir_name.joinpath(f'feature_{i}.png')
        matplotlib.image.imsave(str(w_name), w.reshape(28, 28, order='F'),
                cmap='Greys_r')


# Seed for consistency
np.random.seed(69420)
# File locations
CLASS_MODEL_DIR = CODE_DIR.joinpath('classifier')
CLASS_FEAT_DIR = IMG_DIR.joinpath(f'classifier_features')
CLASS_FEAT_DIR.mkdir(mode=0o775, exist_ok=True)
AUTO_MODEL_DIR = CODE_DIR.joinpath('autoencoder')
AUTO_FEAT_DIR = IMG_DIR.joinpath(f'autoencoder_features')
AUTO_FEAT_DIR.mkdir(mode=0o775, exist_ok=True)
# Load best weights back up for each model
autoencoder = Autoencoder(input_size=INPUTS)
for weight_file in sorted(AUTO_MODEL_DIR.iterdir()):
    autoencoder.addLayer(file_name=weight_file)
classifier = Classifier(input_size=INPUTS)
for weight_file in sorted(CLASS_MODEL_DIR.iterdir()):
    classifier.addLayer(file_name=weight_file)
# Neurons to check
neuron_count = classifier.layers[0].hidden_neurons
neurons = np.random.choice(np.arange(neuron_count), 20, replace=False)
class_weights = classifier.layers[0].w[neurons]
drawFeatures(class_weights, CLASS_FEAT_DIR)
auto_weights = autoencoder.layers[0].w[neurons]
drawFeatures(auto_weights, AUTO_FEAT_DIR)

