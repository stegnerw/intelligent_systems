###############################################################################
# Imports
###############################################################################
# Custom imports
from dataset import shufflePair
from hidden_layer import HiddenLayer
from output_layer import OutputLayer
# External imports
import numpy as np
import enlighten # Progress bar for training

# Manager for the progress bar
PBAR_MANAGER = enlighten.get_manager()


class MLP:
    def __init__(
            self,
            layer_sizes = [784, 128, 10],
        ):
        '''Initialize Multi-Layer Perceptron object
        Parameters:
        -----------
            layer_sizes : list of int
                Number of neurons in each layer
                First element is the size of the input data
        Returns:
            MPL
                The MPL object which was constructed
        '''
        # Create layers
        self.layers = list()
        neurons = layer_sizes[1:]
        inputs = layer_sizes[:-1]
        for idx, (n, i) in enumerate(zip(neurons, inputs)):
            # Check output layer
            if (idx + 1) == len(inputs):
                self.layers.append(OutputLayer(shape=(n,i)))
            else:
                self.layers.append(HiddenLayer(shape=(n,i)))

    def predict(self, data, one_hot=False):
        '''Predict and threshold output of
        Parameters:
        -----------
            data : np.ndarray
                Data point to predict on
            one_hot : bool, optional
                Whether the output should be one-hot or raw
        Returns:
        --------
            np.ndarray
                Array of prediction
        '''
        # Forward pass through all the layers
        layer_output = data
        for l in self.layers:
            layer_output = l.forwardPass(layer_output)
        if one_hot:
            max_idx = np.argmax(layer_output)
            pred = np.eye(10)[max_idx]
        else:
            pred = layer_output
        return pred

    def trainBatch(self, data, labels, eta, alpha, L, H):
        '''Update the weights for a single batch
        For now, a batch is just 1 data point
        Parameters:
        -----------
            data : np.ndarray
                Array of data for the batch
            labels : np.ndarray
                Array of labels for the batch
            eta : float
                Learning rate
            alpha : float
                Momentum scalar
            L, H : float, optional
                Low and high thresholds for training
        Returns:
        --------
            None
        '''
        # Forward pass through all the layers
        self.predict(data)
        self.layers[-1].thresholdOutputs(L, H)
        # Weight change for last layer
        self.layers[-1].setLabel(labels)
        self.layers[-1].getWChange()
        # Back-prop error
        for i in range(len(mlp.layers)-2, -1, -1):
            down_w = self.layers[i+1].w
            down_delta = self.layers[i+1].delta
            self.layers[i].setDownstreamSum(down_w, down_delta)
            self.layers[i].getWChange(eta, alpha)
        # Apply weight changes
        for l in self.layers:
            l.changeW()

    def eval(self, data, labels):
        '''Evaluate classification error on a data set
        Parameters:
        -----------
            data : np.ndarray
                Array of data points
            labels : np.ndarray
                Array of labels for the data
        Returns:
        --------
            float
                Error (1 - Hit Rate)
        '''
        correct = 0
        for d, l in zip(data, labels):
            # Make prediction
            pred = self.predict(d, one_hot=True)
            # Check equality with label
            if (np.sum(np.abs(pred - l)) == 0):
                correct += 1
        hit_rate = correct / len(data)
        return (1 - hit_rate)

    def logError(self,
            train_data,
            train_labels,
            valid_data,
            valid_labels,
            epoch_num,
        ):
        '''Log error for train/test data for a given epoch
        Parameters:
        -----------
            train_data : np.ndarray
                Array of data points for the train set
            train_labels : np.ndarray
                Array of labels for the train set
            valid_data : np.ndarray
                Array of data points for the validation set
            valid_labels : np.ndarray
                Array of labels for the validation set
            epoch_num : int
                The current epoch number
        Returns:
        --------
            None
        '''
        # Evaluate and log train error
        train_err = self.eval(train_data, train_labels)
        self.train_err.append(train_err)
        # Evaluate and log test error
        valid_err = self.eval(valid_data, valid_labels)
        self.valid_err.append(valid_err)
        # Record epoch number
        self.epoch_num.append(epoch_num)
        # # Print out metrics
        # print(f'Epoch {epoch_num}')
        # print(f'\tTrain Error:\t\t{train_err:0.03f}')
        # print(f'\tValidation Error:\t{valid_err:0.03f}')

    def train(
            self,
            data,
            labels,
            points_per_epoch,
            valid_points,
            max_epochs,
            eta,
            alpha,
            L,
            H,
            patience,
            save_dir,
        ):
        '''Train the network up to the desired number of epochs
        Parameters:
        -----------
            data : np.ndarray
                Array of training data points
            labels : np.ndarray
                Array of training labels
            points_per_epoch : int
                Number of training points to use in each epoch
            valid_points : int
                Number of training points to set aside for validation
                These points are set aside before training
            max_epochs : int
                Maximum number of epochs to train
            eta : float
                Learning rate
            alpha : float
                Momentum scalar
            L, H : float
                Low and high thresholds for training
            patience : int, optional
                Amount of epochs with no improvement before early stopping
            save_dir : pathlib.Path or str
                Directory to save best model parameters in
        Returns:
        --------
            None
        '''
        # Initialize progress bar
        pbar = PBAR_MANAGER.counter(total=max_epochs, desc='Training',
                unit = 'epochs', leave=False)
        # Lists to track metrics
        self.train_err = list()
        self.valid_err = list()
        self.epoch_num = list()
        # Set aside validation partition (after shuffling)
        assert valid_points + points_per_epoch <= len(data), \
                'Not enough data for validation points and points per epoch'
        shufflePair(data, labels)
        valid_data = data[:valid_points]
        valid_labels = labels[:valid_points]
        train_data = data[valid_points:]
        train_labels = labels[valid_points:]
        # Log initial accuracy (using whole train partition)
        self.logError(
                train_data,
                train_labels,
                valid_data,
                valid_labels,
                0,
        )
        # Iterate through epochs or until early stopping
        impatience = 0
        self.best_weights_epoch = 0
        for e in range(1, max_epochs+1):
            pbar.update()
            # Shuffle data
            shufflePair(train_data, train_labels)
            epoch_train_data = train_data[:points_per_epoch]
            epoch_train_labels = train_labels[:points_per_epoch]
            for d, l in zip(epoch_train_data, epoch_train_labels):
                self.trainBatch(d, l, eta, alpha, L, H)
            # Log data every 10 epochs
            if (e % 10) == 0:
                self.logError(
                        epoch_train_data,
                        epoch_train_labels,
                        valid_data,
                        valid_labels,
                        e,
                )
                # Check for early stopping
                if (min(self.valid_err) == self.valid_err[-1]):
                    # print('\t* * * Saving new best weights * * *')
                    impatience = 0
                    for i, l in enumerate(self.layers):
                        layer_name = save_dir.joinpath(f'layer_{i:02d}')
                        l.saveWeights(layer_name)
                else:
                    impatience += 1
                    # We have become too impatient
                    if impatience >= patience:
                        # print(f'* * * Early stopping hit * * *')
                        break
        # TODO - Save training specs including epochs trained
        pbar.close()

    def makeConfMat(self, data, labels, plot_name, title='Confusion Matrix'):
        '''Generate and save a confusion matrix
        Parameters:
        -----------
            data : np.ndarray
                Array of data values
            labels : np.ndarray
                Array of labels as one-hot-vectors
            plot_name : pathlib.Path or str
                File name to save the matrix as
        Returns:
        --------
            None
        '''
        assert len(data) == len(labels), \
                'Size mismatch between data and labels'
        num_classes = len(labels[0])
        conf_mat = np.zeros((num_classes, num_classes))
        for d, l in zip(data, labels):
            pred = self.predict(d, one_hot=True)
            class_count = len(l)
            conf_mat += l.reshape((class_count,1)) \
                        * pred.reshape((1,class_count))
        # Plot confusion matrix and save
        plt.figure()
        plt.suptitle('Confusion Matrix')
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


if __name__ == '__main__':
    # Seed for consistency
    np.random.seed(69420)
    # Additional imports
    import pathlib
    import pickle
    import matplotlib.pyplot as plt
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
    PLT_NAME = IMG_DIR.joinpath('mlp_accuracy.png')
    CONF_NAME = IMG_DIR.joinpath('conf_mat.png')
    MODELS_DIR = CODE_DIR.joinpath('models')
    MODELS_DIR.mkdir(mode=0o775, exist_ok=True)
    # Training constants
    POINTS_PER_EPOCH = 1000
    VALID_POINTS = 1000
    MAX_EPOCHS=1000
    ETA = 0.1
    ALPHA = 0.8
    L = 0.25
    H = 0.75
    PATIENCE = 5
    # GRID SEARCH BOIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
    grid_pbar = PBAR_MANAGER.counter(total=90, desc='Grid Search',
            unit = 'trial')
    min_err = 1
    errors = list()
    for ETA in [0.5, 0.1, 0.05, 0.01, 0.005]:
        for ALPHA in [0, 0.5, 0.8, 0.9, 0.95, 0.99]:
            for HIDDEN_NEURONS in [64, 128, 192]:
                grid_pbar.update()
                # Define model directory and clear existing weight files
                MODEL_DIR = MODELS_DIR.joinpath(f'{ETA}_eta_{ALPHA}_alpha')
                PLT_NAME = IMG_DIR.joinpath(f'mlp_accuracy_{ETA}_eta_{ALPHA}_alpha.png')
                CONF_NAME = IMG_DIR.joinpath(f'conf_mat_{ETA}_eta_{ALPHA}_alpha.png')
                MODEL_DIR.mkdir(mode=0o775, exist_ok=True)
                for f in MODEL_DIR.iterdir():
                    f.unlink()
                # Load dataset from files
                train_data = np.load(str(TRAIN_DATA_FILE))
                train_labels = np.load(str(TRAIN_LABELS_FILE))
                test_data = np.load(str(TEST_DATA_FILE))
                test_labels = np.load(str(TEST_LABELS_FILE))
                # Test network
                mlp = MLP(layer_sizes = [784, HIDDEN_NEURONS, 10])
                # Train the network
                mlp.train(
                    train_data,
                    train_labels,
                    points_per_epoch=POINTS_PER_EPOCH,
                    valid_points=VALID_POINTS,
                    max_epochs=MAX_EPOCHS,
                    eta=ETA,
                    alpha=ALPHA,
                    L=L,
                    H=H,
                    patience=PATIENCE,
                    save_dir=MODEL_DIR,
                )
                plt.figure()
                plt.plot(mlp.epoch_num, mlp.train_err)
                plt.plot(mlp.epoch_num, mlp.valid_err)
                plt.legend(['Train', 'Validation'], loc='upper right')
                plt.title(f'Error vs Epoch Number @ Eta = {ETA}, Alpha = {ALPHA}')
                plt.xlabel('Epoch number')
                plt.ylabel('Error')
                plt.xlim([0, mlp.epoch_num[-1]])
                plt.ylim([0, max(mlp.train_err[0], mlp.valid_err[0])])
                plt.savefig(str(PLT_NAME))
                plt.close()
                conf_title = f'Confusion Matrix @ Eta = {ETA}, Alpha = {ALPHA}'
                mlp.makeConfMat(test_data, test_labels, CONF_NAME, title=conf_title)
                test_err = mlp.eval(test_data, test_labels)
                # Check for new minimum error
                if test_err < min_err:
                    min_err = test_err
                    print(f'New best error:\t{min_err}')
                    print(f'\tEta:\t\t{ETA}')
                    print(f'\tAlpha:\t\t{ALPHA}')
                    print(f'\tHidden neurons:\t{HIDDEN_NEURONS}')
                errors.append({
                    'eta': ETA,
                    'alpha': ALPHA,
                    'hidden_neurons': HIDDEN_NEURONS,
                    'error': test_err,
                })
    PBAR_MANAGER.stop()
    # Write csv
    errors.sort(key=lambda x: x['error'])
    import csv
    col_names = ['eta', 'alpha', 'hidden_neurons', 'error']
    with open('grid_searchy_boi.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=col_names)
        writer.writeheader()
        for error in errors:
            writer.writerow(error)

