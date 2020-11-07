###############################################################################
# Imports
###############################################################################
import pathlib
import numpy as np


###############################################################################
# Constant Values
###############################################################################
# Common constants
TRAIN_PORTION = 0.8
SEED = 69420
INPUTS = 784
HIDDEN_LAYER_SIZES = [192]
CLASSES = 10
# Clean classifier constants
CLASS_CLEAN_POINTS_PER_EPOCH = 1000
CLASS_CLEAN_VALID_POINTS = 1000
CLASS_CLEAN_MAX_EPOCHS = 1000
CLASS_CLEAN_ETA = 0.05
CLASS_CLEAN_ALPHA = 0.8
CLASS_CLEAN_L = 0.25
CLASS_CLEAN_H = 0.75
CLASS_CLEAN_PATIENCE = 10
CLASS_CLEAN_ES_DELTA = 0.01
# Clean autoencoder constants
AUTO_CLEAN_POINTS_PER_EPOCH = 1000
AUTO_CLEAN_VALID_POINTS = 1000
AUTO_CLEAN_MAX_EPOCHS = 1000
AUTO_CLEAN_ETA = 0.0025
AUTO_CLEAN_ALPHA = 0.8
AUTO_CLEAN_L = 0
AUTO_CLEAN_H = 1
AUTO_CLEAN_PATIENCE = 10
AUTO_CLEAN_ES_DELTA = 0.1
# Noise parameters
SALT_PROB = 0.05
PEP_PROB = 0.4
# Noisy classifier constants
CLASS_NOISY_POINTS_PER_EPOCH = 1000
CLASS_NOISY_VALID_POINTS = 1000
CLASS_NOISY_MAX_EPOCHS = 1000
CLASS_NOISY_ETA = 0.025
CLASS_NOISY_ALPHA = 0.8
CLASS_NOISY_L = 0.25
CLASS_NOISY_H = 0.75
CLASS_NOISY_PATIENCE = 10
CLASS_NOISY_ES_DELTA = 0.01
# Noisy autoencoder constants
AUTO_NOISY_POINTS_PER_EPOCH = 1000
AUTO_NOISY_VALID_POINTS = 1000
AUTO_NOISY_MAX_EPOCHS = 1000
AUTO_NOISY_ETA = 0.0025
AUTO_NOISY_ALPHA = 0.8
AUTO_NOISY_L = 0
AUTO_NOISY_H = 1
AUTO_NOISY_PATIENCE = 10
AUTO_NOISY_ES_DELTA = 0.1


###############################################################################
# File locations
###############################################################################
# Directory of current file
CODE_DIR = pathlib.Path(__file__).parent.absolute()
# Root directory of the project
ROOT_DIR = CODE_DIR.parent
# Dataset locations
DATA_FILE = CODE_DIR.joinpath('data.txt')
LABEL_FILE = CODE_DIR.joinpath('labels.txt')
DATASET_DIR = CODE_DIR.joinpath('dataset')
DATASET_DIR.mkdir(mode=0o755, exist_ok=True)
TRAIN_DATA_FILE = DATASET_DIR.joinpath('train_data.npy')
TRAIN_LABELS_FILE = DATASET_DIR.joinpath('train_labels.npy')
TEST_DATA_FILE = DATASET_DIR.joinpath('test_data.npy')
TEST_LABELS_FILE = DATASET_DIR.joinpath('test_labels.npy')
# Image directory
IMG_DIR = ROOT_DIR.joinpath('images')
IMG_DIR.mkdir(mode=0o775, exist_ok=True)
# Data directory
DATA_DIR = ROOT_DIR.joinpath('data')
DATA_DIR.mkdir(mode=0o775, exist_ok=True)
# Clean classifier locations
CLASS_CLEAN_NAME = 'class_clean'
CLASS_CLEAN_MODEL_DIR = CODE_DIR.joinpath(CLASS_CLEAN_NAME)
CLASS_CLEAN_MODEL_DIR.mkdir(mode=0o775, exist_ok=True)
CLASS_CLEAN_IMG_DIR = IMG_DIR.joinpath(CLASS_CLEAN_NAME)
CLASS_CLEAN_IMG_DIR.mkdir(mode=0o775, exist_ok=True)
CLASS_CLEAN_LOSS_PLOT = CLASS_CLEAN_IMG_DIR.joinpath('loss_graph.png')
CLASS_CLEAN_TRAIN_CONF = CLASS_CLEAN_IMG_DIR.joinpath('train_conf_mat.png')
CLASS_CLEAN_TEST_CONF = CLASS_CLEAN_IMG_DIR.joinpath('test_conf_mat.png')
CLASS_CLEAN_DATA_DIR = DATA_DIR.joinpath(CLASS_CLEAN_NAME)
CLASS_CLEAN_DATA_DIR.mkdir(mode=0o775, exist_ok=True)
CLASS_CLEAN_TEST_LOSS = CLASS_CLEAN_DATA_DIR.joinpath('test_loss.dat')
CLASS_CLEAN_PARAM_CSV = CLASS_CLEAN_DATA_DIR.joinpath('parameters.csv')
CLASS_CLEAN_BEST_EPOCH = CLASS_CLEAN_DATA_DIR.joinpath('best_epoch.dat')
# Clean autoencoder files
AUTO_CLEAN_NAME = 'auto_clean'
AUTO_CLEAN_MODEL_DIR = CODE_DIR.joinpath(AUTO_CLEAN_NAME)
AUTO_CLEAN_MODEL_DIR.mkdir(mode=0o775, exist_ok=True)
AUTO_CLEAN_IMG_DIR = IMG_DIR.joinpath(AUTO_CLEAN_NAME)
AUTO_CLEAN_IMG_DIR.mkdir(mode=0o775, exist_ok=True)
AUTO_CLEAN_LOSS_PLOT = AUTO_CLEAN_IMG_DIR.joinpath('loss_graph.png')
AUTO_CLEAN_BAR = AUTO_CLEAN_IMG_DIR.joinpath('loss_bar_plot.png')
AUTO_CLEAN_DATA_DIR = DATA_DIR.joinpath(AUTO_CLEAN_NAME)
AUTO_CLEAN_DATA_DIR.mkdir(mode=0o775, exist_ok=True)
AUTO_CLEAN_TEST_LOSS = AUTO_CLEAN_DATA_DIR.joinpath('test_loss.dat')
AUTO_CLEAN_PARAM_CSV = AUTO_CLEAN_DATA_DIR.joinpath('parameters.csv')
AUTO_CLEAN_BEST_EPOCH = AUTO_CLEAN_DATA_DIR.joinpath('best_epoch.dat')
# Noisy classifier locations
CLASS_NOISY_NAME = 'class_noisy'
CLASS_NOISY_MODEL_DIR = CODE_DIR.joinpath(CLASS_NOISY_NAME)
CLASS_NOISY_MODEL_DIR.mkdir(mode=0o775, exist_ok=True)
CLASS_NOISY_IMG_DIR = IMG_DIR.joinpath(CLASS_NOISY_NAME)
CLASS_NOISY_IMG_DIR.mkdir(mode=0o775, exist_ok=True)
CLASS_NOISY_LOSS_PLOT = IMG_DIR.joinpath('loss_graph.png')
CLASS_NOISY_TRAIN_CONF = IMG_DIR.joinpath('train_conf_mat.png')
CLASS_NOISY_TEST_CONF = IMG_DIR.joinpath('test_conf_mat.png')
CLASS_NOISY_DATA_DIR = DATA_DIR.joinpath(CLASS_NOISY_NAME)
CLASS_NOISY_DATA_DIR.mkdir(mode=0o775, exist_ok=True)
CLASS_NOISY_TEST_LOSS = DATA_DIR.joinpath('test_loss.dat')
CLASS_NOISY_PARAM_CSV = DATA_DIR.joinpath('parameters.csv')
CLASS_NOISY_BEST_EPOCH = DATA_DIR.joinpath('best_epoch.dat')
# Noisy autoencoder files
AUTO_NOISY_NAME = 'auto_noisy'
AUTO_NOISY_MODEL_DIR = CODE_DIR.joinpath(AUTO_NOISY_NAME)
AUTO_NOISY_MODEL_DIR.mkdir(mode=0o775, exist_ok=True)
AUTO_NOISY_IMG_DIR = IMG_DIR.joinpath(AUTO_NOISY_NAME)
AUTO_NOISY_IMG_DIR.mkdir(mode=0o775, exist_ok=True)
AUTO_NOISY_LOSS_PLOT = AUTO_NOISY_IMG_DIR.joinpath('loss_graph.png')
AUTO_NOISY_BAR = AUTO_NOISY_IMG_DIR.joinpath('loss_bar_plot.png')
AUTO_NOISY_DATA_DIR = DATA_DIR.joinpath(AUTO_NOISY_NAME)
AUTO_NOISY_DATA_DIR.mkdir(mode=0o775, exist_ok=True)
AUTO_NOISY_TEST_LOSS = AUTO_NOISY_DATA_DIR.joinpath('test_loss.dat')
AUTO_NOISY_PARAM_CSV = AUTO_NOISY_DATA_DIR.joinpath('parameters.csv')
AUTO_NOISY_BEST_EPOCH = AUTO_NOISY_DATA_DIR.joinpath('best_epoch.dat')

###############################################################################
# Load Dataset
###############################################################################
def noiseData(data):
    """Return a noisy data point from the given clean point
    Currently just does salt and pepper noise
    Parameters
    ----------
    data : np.ndarray
        Data point to be noised
    Returns
    -------
    np.ndarray
        The noisy image
    """
    noisy_data = data.copy()
    salt_pepper = np.random.random(size=data.size)
    noisy_data[salt_pepper > (1 - SALT_PROB)] = 1.0
    noisy_data[salt_pepper < PEP_PROB] = 0.0
    return noisy_data

if TRAIN_DATA_FILE.exists():
    train_data = np.load(str(TRAIN_DATA_FILE))
    noisy_train_data = list()
    for d in train_data:
        noisy_train_data.append(noiseData(d))
    noisy_train_data = np.array(noisy_train_data)
if TRAIN_LABELS_FILE.exists():
    train_labels = np.load(str(TRAIN_LABELS_FILE))
if TEST_DATA_FILE.exists():
    test_data = np.load(str(TEST_DATA_FILE))
    noisy_test_data = list()
    for d in test_data:
        noisy_test_data.append(noiseData(d))
    noisy_test_data = np.array(noisy_test_data)
if TEST_LABELS_FILE.exists():
    test_labels = np.load(str(TEST_LABELS_FILE))

