###############################################################################
# Imports
###############################################################################
# Custom imports
from settings import *
from classifier import Classifier
# External imports
import numpy as np
import pathlib
import matplotlib.pyplot as plt


def makeConfMat(classifier, data, labels, plot_name, title):
    '''Generate and save a confusion matrix
    Parameters
    ----------
    classifier : Classifier
        Classifier for use in classification
    data : np.ndarray
        Array of data values
    labels : np.ndarray
        Array of labels as one-hot-vectors
    plot_name : pathlib.Path or str
        File name to save the matrix as
    title : str
        Title of the confusion matrix
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
    plt.xticks(range(CLASSES))
    plt.ylabel('Actual Value')
    plt.yticks(range(CLASSES))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(str(plot_name), bbox_inches='tight', pad_inches=0)
    plt.close()


# Seed for consistency
np.random.seed(SEED)
# Load best weights back up and make confusion matrices
classifier = Classifier(input_size=INPUTS)
weight_files = sorted(CLASS_NOISY_MODEL_DIR.iterdir())
for weight_file in weight_files[:-1]:
    classifier.addLayer(file_name=weight_file, output=False)
classifier.addLayer(file_name=weight_files[-1], output=True)
train_conf_title = 'Train Confusion Matrix'
makeConfMat(classifier, train_data, train_labels, CLASS_NOISY_TRAIN_CONF,
        title=train_conf_title)
test_conf_title = 'Test Confusion Matrix'
makeConfMat(classifier, test_data, test_labels, CLASS_NOISY_TEST_CONF,
        title=test_conf_title)
test_err = classifier.eval(test_data, test_labels)
print(f'Test error: {test_err:0.3f}')
with open(str(CLASS_NOISY_TEST_LOSS), 'w') as loss_f:
    loss_f.write(f'{test_err:0.3f}')

