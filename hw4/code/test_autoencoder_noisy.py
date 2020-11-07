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
    '''Split up dataset by class
    Parameters
    ----------
    data, labels : np.ndarray
        Arrays of data points and labels
    Returns
    -------
    list
        Data split up by class
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
    Parameters
    ----------
    autoencoder : Autoencoder
        Autoencoder for use in loss calculation
    data, labels : np.ndarray
        Arrays of data points and labels
    Returns
    -------
    float
        Loss values by class
    '''
    loss = list()
    split_data = splitClasses(data, labels)
    for i, d in enumerate(split_data):
        print(f'Evaluating class {i}')
        loss.append(autoencoder.eval(d, d))
    return loss

def getSamplePoints(data, n):
    '''Get sample points from the given data set
    Parameters
    ----------
    data : np.ndarray
        Array of data points
    n : int
        Number of sample points
    Returns
    -------
    np.ndarray
        Reduced list of data points
    '''
    indeces = np.random.choice(np.arange(len(data)), 8, replace=False)
    return data[indeces]

def drawSamples(autoencoder, data, num_samples, dir_name, title):
    '''Draw the output predictions and save them
    Parameters
    ----------
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
np.random.seed(SEED)
# Load best weights back up
autoencoder = Autoencoder(input_size=INPUTS)
weight_files = sorted(AUTO_NOISY_MODEL_DIR.iterdir())
for weight_file in weight_files[:-1]:
    autoencoder.addLayer(file_name=weight_file, output=False)
autoencoder.addLayer(file_name=weight_files[-1], output=True)
# Test on all data and draw samples
test_err = autoencoder.eval(noisy_test_data, test_data)
print(f'Test loss: {test_err:0.3f}')
sample_title = 'Autoencoder Sample Outputs'
drawSamples(autoencoder, noisy_test_data, 8, AUTO_NOISY_IMG_DIR, sample_title)
# Graph loss by class
print('Testing train set')
train_loss = getLossByClass(autoencoder, noisy_train_data, train_labels)
print('Testing test set')
test_loss = getLossByClass(autoencoder, noisy_test_data, test_labels)
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
plt.savefig(str(AUTO_NOISY_BAR), bbox_inches='tight', pad_inches=0)
with open(str(AUTO_NOISY_TEST_LOSS), 'w') as loss_f:
    loss_f.write(f'{test_err:0.3f}')

