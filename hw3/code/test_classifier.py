###############################################################################
# Imports
###############################################################################
# Custom imports
from classifier import Classifier
from settings import *
# External imports
import numpy as np
import pathlib
import matplotlib.pyplot as plt


def makeConfMat(classifier, data, labels, plot_name, title):
    '''Generate and save a confusion matrix
    Parameters:
    -----------
        data : np.ndarray
            Array of data values
        labels : np.ndarray
            Array of labels as one-hot-vectors
        plot_name : pathlib.Path or str
            File name to save the matrix as
        title : str
            Title of the confusion matrix
    Returns:
    --------
        None
    '''
    assert len(data) == len(labels), \
            'Size mismatch between data and labels'
    conf_mat = np.zeros((CLASSES, CLASSES))
    for d, l in zip(data, labels):
        pred = classifier.predict(d, one_hot=True)
        conf_mat += l.reshape((CLASSES,1)) * pred.reshape((1,CLASSES))
    # Plot confusion matrix and save
    plt.figure()
    plt.suptitle(title)
    plt.imshow(conf_mat, cmap='Greens')
    for i in range(len(conf_mat)):
        for j in range(len(conf_mat[i])):
            color = 'k' if (conf_mat[i][j] <= 50) else 'w'
            plt.text(j, i, f'{int(conf_mat[i][j])}',
                    va='center', ha='center', color=color)
    plt.xlabel('Predicted Value')
    plt.xticks(range(num_classes))
    plt.ylabel('Actual Value')
    plt.yticks(range(num_classes))
    plt.colorbar()
    plt.savefig(str(plot_name))
    plt.close()


# Seed for consistency
np.random.seed(69420)
# File locations
TRAIN_CONF_NAME = IMG_DIR.joinpath(f'train_conf_mat.png')
TEST_CONF_NAME = IMG_DIR.joinpath(f'test_conf_mat.png')
MODEL_DIR = CODE_DIR.joinpath('classifier')
# Load best weights back up and make confusion matrices
classifier = Classifier(input_size=INPUTS)
for weight_file in sorted(MODEL_DIR.iterdir()):
    classifier.addLayer(file_name=weight_file)
train_conf_title = 'Train Confusion Matrix'
makeConfMat(classifier, train_data, train_labels, TRAIN_CONF_NAME,
        title=train_conf_title)
test_conf_title = 'Test Confusion Matrix'
makeConfMat(classifier, test_data, test_labels, TEST_CONF_NAME,
        title=test_conf_title)
test_err = classifier.eval(test_data, test_labels)
print(f'Test error: {test_err:0.3f}')

