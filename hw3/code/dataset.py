###############################################################################
# Imports
###############################################################################
from settings import *
import numpy as np
import pathlib
import matplotlib.pyplot as plt


def shufflePair(data, labels):
    '''Shuffle a pair of data and labels in place
    Parameters:
    -----------
        data, labels : np.ndarray
            Data and labels to be shuffled
    Returns:
    --------
        None
    '''
    assert len(data) == len(labels), \
            'Size mismatch between data and labels'
    indeces = np.random.permutation(len(data))
    data[...] = data[indeces]
    labels[...] = labels[indeces]


if __name__ == '__main__':
    # # Define important values from settings
    # SEED = 69420
    # CODE_DIR = pathlib.Path(__file__).parent.absolute()
    # DATA_FILE = CODE_DIR.joinpath('data.txt')
    # LABEL_FILE = CODE_DIR.joinpath('labels.txt')
    # DATASET_DIR = CODE_DIR.joinpath('dataset')
    # DATASET_DIR.mkdir(mode=0o755, exist_ok=True)
    # TRAIN_DATA_FILE = DATASET_DIR.joinpath('train_data.npy')
    # TRAIN_LABELS_FILE = DATASET_DIR.joinpath('train_labels.npy')
    # TEST_DATA_FILE = DATASET_DIR.joinpath('test_data.npy')
    # TEST_LABELS_FILE = DATASET_DIR.joinpath('test_labels.npy')
    # Seed for consistency
    np.random.seed(SEED)
    ###########################################################################
    # Import dataset
    ###########################################################################
    # Check that the files exist
    assert DATA_FILE.exists(), f'File not found: {str(data_file)}'
    assert LABEL_FILE.exists(), f'File not found: {str(label_file)}'
    # Read data and labels from txt file
    data = list()
    with open(DATA_FILE) as data_f:
        data = data_f.readlines()
    data = [d.split() for d in data]
    labels = list()
    with open(LABEL_FILE) as label_f:
        labels = label_f.readlines()
    labels = [int(l) for l in labels]
    # Make images and sort into bins based on class
    data_points = dict()
    classes = list()
    for d, l in zip(data, labels):
        if l not in classes:
            classes.append(l)
            data_points[l] = list()
        data_points[l].append(d)
    classes.sort()
    # Turn data lists into numpy arrays
    for l in classes:
        data_points[l] = np.array(data_points[l], dtype=np.float64)
    ###########################################################################
    # Partition dataset
    ###########################################################################
    train_data = list()
    train_labels = list()
    test_data = list()
    test_labels = list()
    # Shuffle the data points first
    for l in classes:
        np.random.shuffle(data_points[l])
    # Iterate through the data by class and partition
    points_per_class = len(data_points[classes[0]])
    train_per_class = int(TRAIN_PORTION * points_per_class)
    for l in classes:
        for i, d in enumerate(data_points[l]):
            if i < train_per_class:
                train_data.append(d)
                train_labels.append(l)
            else:
                test_data.append(d)
                test_labels.append(l)
    # Turn lists into numpy arrays
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    # Turn labels into one-hot arrays
    train_labels = np.eye(len(classes))[train_labels]
    test_labels = np.eye(len(classes))[test_labels]
    # Shuffle arrays
    shufflePair(train_data, train_labels)
    shufflePair(test_data, test_labels)
    ###########################################################################
    # Save arrays
    ###########################################################################
    np.save(str(TRAIN_DATA_FILE), train_data)
    np.save(str(TRAIN_LABELS_FILE), train_labels)
    np.save(str(TEST_DATA_FILE), test_data)
    np.save(str(TEST_LABELS_FILE), test_labels)

