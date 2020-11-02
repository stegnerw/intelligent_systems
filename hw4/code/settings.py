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
# Classifier constants
CLASS_POINTS_PER_EPOCH = 1000
CLASS_VALID_POINTS = 1000
CLASS_MAX_EPOCHS = 500
CLASS_ETA = 0.05
CLASS_ALPHA = 0.8
CLASS_L = 0.25
CLASS_H = 0.75
CLASS_PATIENCE = 3
CLASS_ES_DELTA = 0.01
# Autoencoder constants
AUTO_POINTS_PER_EPOCH = 1000
AUTO_VALID_POINTS = 1000
AUTO_MAX_EPOCHS = 500
AUTO_ETA = 0.005
AUTO_ALPHA = 0.8
AUTO_L = 0
AUTO_H = 1
AUTO_PATIENCE = 3
AUTO_ES_DELTA = 0.1


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
# Classifier locations
CLASS_MODEL_DIR = CODE_DIR.joinpath('class')
CLASS_MODEL_DIR.mkdir(mode=0o775, exist_ok=True)
CLASS_LOSS_PLOT = IMG_DIR.joinpath('class_loss.png')
CLASS_PARAM_CSV = DATA_DIR.joinpath('class_parameters.csv')
CLASS_BEST_EPOCH = DATA_DIR.joinpath('class_best_epoch.dat')
CLASS_TRAIN_CONF = IMG_DIR.joinpath('train_conf_mat.png')
CLASS_TEST_CONF = IMG_DIR.joinpath('test_conf_mat.png')
CLASS_TEST_LOSS = DATA_DIR.joinpath('class_test_loss.dat')
CLASS_FEAT_DIR = IMG_DIR.joinpath('class_feat')
CLASS_FEAT_DIR.mkdir(mode=0o775, exist_ok=True)
# Autoencoder files
AUTO_MODEL_DIR = CODE_DIR.joinpath('auto')
AUTO_MODEL_DIR.mkdir(mode=0o775, exist_ok=True)
AUTO_LOSS_PLOT = IMG_DIR.joinpath('auto_loss.png')
AUTO_PARAM_CSV = DATA_DIR.joinpath('auto_parameters.csv')
AUTO_BEST_EPOCH = DATA_DIR.joinpath('auto_best_epoch.dat')
AUTO_BAR = IMG_DIR.joinpath('loss_bar_plot.png')
AUTO_SAMPLE_DIR = IMG_DIR.joinpath('auto_samples')
AUTO_SAMPLE_DIR.mkdir(mode=0o775, exist_ok=True)
AUTO_TEST_LOSS = DATA_DIR.joinpath('auto_test_loss.dat')
AUTO_FEAT_DIR = IMG_DIR.joinpath('auto_feat')
AUTO_FEAT_DIR.mkdir(mode=0o775, exist_ok=True)

###############################################################################
# Load Dataset
###############################################################################
if TRAIN_DATA_FILE.exists():
    train_data = np.load(str(TRAIN_DATA_FILE))
if TRAIN_LABELS_FILE.exists():
    train_labels = np.load(str(TRAIN_LABELS_FILE))
if TEST_DATA_FILE.exists():
    test_data = np.load(str(TEST_DATA_FILE))
if TEST_LABELS_FILE.exists():
    test_labels = np.load(str(TEST_LABELS_FILE))

