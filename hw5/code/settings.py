###############################################################################
# Imports
###############################################################################
import pathlib
import cupy as np


###############################################################################
# Constant Values
###############################################################################
# Common constants
TRAIN_PORTION = 0.8
SEED = 69420
INPUTS = 784
CLASSES = 10
# Classifier constants
CLASS_POINTS_PER_EPOCH = 3000
CLASS_VALID_POINTS = 1000
CLASS_MAX_EPOCHS = 1000
CLASS_ETA = 0.01
CLASS_ALPHA = 0.8
CLASS_DECAY = 0.0001
CLASS_L = 0.25
CLASS_H = 0.75
CLASS_PATIENCE = 10
CLASS_ES_DELTA = 0.0
# SOFM constants
SOFM_SHAPE = (12, 12)
SOFM_ETA_0 = 0.1 # Initial eta
SOFM_ETA_FLOOR = 0.01 # Stage 2 eta (final)
SOFM_SIGMA_0 = 0.5 * max(SOFM_SHAPE) # Initial sigma
SOFM_SIGMA_FLOOR = 0.1 # Stage 2 sigma (final)
SOFM_PHASE_1_EPOCHS = 1000
SOFM_PHASE_2_EPOCHS = 500 * SOFM_SHAPE[0] * SOFM_SHAPE[1]
SOFM_MAX_EPOCHS = SOFM_PHASE_1_EPOCHS + SOFM_PHASE_2_EPOCHS
SOFM_TAU_L = -SOFM_PHASE_1_EPOCHS / np.log(SOFM_ETA_FLOOR / SOFM_ETA_0)
SOFM_TAU_N = -SOFM_PHASE_1_EPOCHS / np.log(SOFM_SIGMA_FLOOR / SOFM_SIGMA_0)

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
CLASS_NAME = 'class'
CLASS_MODEL_DIR = CODE_DIR.joinpath(CLASS_NAME)
CLASS_MODEL_DIR.mkdir(mode=0o775, exist_ok=True)
CLASS_LOSS_PLOT = IMG_DIR.joinpath('loss_graph.png')
CLASS_TRAIN_CONF = IMG_DIR.joinpath('train_conf_mat.png')
CLASS_TEST_CONF = IMG_DIR.joinpath('test_conf_mat.png')
CLASS_TEST_LOSS = DATA_DIR.joinpath('test_loss.dat')
CLASS_PARAM_CSV = DATA_DIR.joinpath('parameters.csv')
CLASS_BEST_EPOCH = DATA_DIR.joinpath('best_epoch.dat')
# SOFM locations
SOFM_NAME = 'sofm'
SOFM_MODEL_DIR = CODE_DIR.joinpath(SOFM_NAME)
SOFM_MODEL_DIR.mkdir(mode=0o755, exist_ok=True)
SOFM_WEIGHT_FILE = SOFM_MODEL_DIR.joinpath('layer_00.npy')
SOFM_FEAT = IMG_DIR.joinpath('features.png')
SOFM_HEATMAP_DIR = IMG_DIR.joinpath('heatmaps')
SOFM_HEATMAP_DIR.mkdir(mode=0o755, exist_ok=True)
SOFM_PARAM_CSV = DATA_DIR.joinpath('parameters.csv')

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

