###############################################################################
# Imports
###############################################################################
# Custom imports
from autoencoder import Autoencoder
from settings import *
# External imports
import numpy as np
import pathlib
import matplotlib
import matplotlib.pyplot as plt


def splitClasses(data, labels):
    '''
    '''
    split_data = list()
    for i in range(CLASSES):
        split_data.append(list())
    for d, l in zip(data, labels):
        idx = np.argmax(l)
        split_data[idx].append(d)
    return split_data

def getLossByClass(autoencoder, data, labels):
    '''
    '''
    loss = list()
    split_data = splitClasses(data, labels)
    for i, d in enumerate(split_data):
        print(f'Evaluating class {i}')
        loss.append(autoencoder.eval(d, d))
    return loss

def getSamplePoints(data, n):
    '''Get sample points from the given data set
    Parameters:
    -----------
        data : np.ndarray
            Array of data points
        n : int
            Number of sample points
    Returns:
    --------
        np.ndarray
            Reduced list of data points
    '''
    indeces = np.random.choice(np.arange(len(data)), 8, replace=False)
    return data[indeces]

def drawSamples(autoencoder, data, num_samples, dir_name, title):
    '''Draw the output predictions and save them
    Parameters:
    -----------
        autoencoder : Autoencoder
            Autoencoder for use in inference
        data : np.ndarray
            Array of data points
        num_samples : int
            Number of sample points
        dir_name : str
            Name of the directory to save the images to
        title : str
            Title of the plot
    '''
    sample_points = getSamplePoints(data, num_samples)
    for i, d in enumerate(sample_points):
        d_name = dir_name.joinpath(f'orig_{i}.png')
        matplotlib.image.imsave(str(d_name), d.reshape(28, 28, order='F'), cmap='Greys_r')
        pred = autoencoder.predict(d, one_hot=False)
        p_name = dir_name.joinpath(f'pred_{i}.png')
        matplotlib.image.imsave(str(p_name), pred.reshape(28, 28, order='F'), cmap='Greys_r')


# Seed for consistency
np.random.seed(69420)
# File locations
SAMPLE_NAME = IMG_DIR.joinpath(f'sample_points.png')
BAR_NAME = IMG_DIR.joinpath(f'loss_bar_plot.png')
SAMPLE_DIR = IMG_DIR.joinpath(f'autoencoder_samples')
SAMPLE_DIR.mkdir(mode=0o775, exist_ok=True)
MODEL_DIR = CODE_DIR.joinpath('autoencoder')
# Load best weights back up
autoencoder = Autoencoder(input_size=INPUTS)
for weight_file in sorted(MODEL_DIR.iterdir()):
    autoencoder.addLayer(file_name=weight_file)
# Test on all data and draw samples
test_err = autoencoder.eval(test_data, test_data)
print(f'Test loss: {test_err:0.3f}')
sample_title = 'Autoencoder Sample Outputs'
drawSamples(autoencoder, test_data, 8, SAMPLE_DIR, sample_title)
# Graph loss by class
train_loss = getLossByClass(autoencoder, train_data, train_labels)
test_loss = getLossByClass(autoencoder, test_data, test_labels)
x = np.arange(len(train_loss))
plt.figure()
rect_width = 0.35
plt.bar(x-rect_width/2, train_loss, rect_width, label='Train')
plt.bar(x+rect_width/2, test_loss, rect_width, label='Test')
plt.title('Autoencoder Loss by Class')
plt.xlabel('Class')
plt.xticks(x)
plt.ylabel('Loss')
plt.grid(axis='y')
plt.gca().set_axisbelow(True)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(str(BAR_NAME))

