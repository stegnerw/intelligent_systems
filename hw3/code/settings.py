###############################################################################
# Imports
###############################################################################
# External imports
import pathlib
import numpy as np


# File locations
CODE_DIR = pathlib.Path(__file__).parent.absolute()
DATASET_DIR = CODE_DIR.joinpath('dataset')
TRAIN_DATA_FILE = DATASET_DIR.joinpath('train_data.npy')
TRAIN_LABELS_FILE = DATASET_DIR.joinpath('train_labels.npy')
TEST_DATA_FILE = DATASET_DIR.joinpath('test_data.npy')
TEST_LABELS_FILE = DATASET_DIR.joinpath('test_labels.npy')
ROOT_DIR = CODE_DIR.parent
DATA_FILE = CODE_DIR.joinpath('data.txt')
LABEL_FILE = CODE_DIR.joinpath('labels.txt')
IMG_DIR = ROOT_DIR.joinpath('images')
IMG_DIR.mkdir(mode=0o775, exist_ok=True)
# Load dataset from files
train_data = np.load(str(TRAIN_DATA_FILE))
train_labels = np.load(str(TRAIN_LABELS_FILE))
test_data = np.load(str(TEST_DATA_FILE))
test_labels = np.load(str(TEST_LABELS_FILE))
# Constant values
INPUTS = 784
HIDDEN_NEURONS = 192
CLASSES = 10

