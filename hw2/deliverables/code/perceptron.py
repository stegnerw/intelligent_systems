###############################################################################
# Imports
###############################################################################
from dataset import Dataset
from classifier import Classifier
import numpy as np
from matplotlib import pyplot as plt


class Perceptron(Classifier):
    def __init__(
            self,
            dataset,
            num_inputs = 2,
            learning_rate = 0.1,
    ):
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.init_weights()
        self.dataset = dataset
        self.resetStats()
        self.resetAcc()

    def resetAcc(self):
        '''Reset accuracy metrics
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        '''
        self.train_acc = list()
        self.test_acc = list()
        self.epoch_nums = list()

    def init_weights(self):
        '''Initialize weights for the perceptron
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        '''
        self.weights = np.empty(self.num_inputs + 1, dtype=np.float64)
        # Initialize bias weight
        self.weights[0] = np.random.uniform(-10, -50)
        # Initialize input weights
        for i in range(len(self.weights) - 1):
            self.weights[i+1] = np.random.uniform(2, 5)

    def get_weights(self):
        '''Return weights array
        Parameters:
        -----------
            None
        Returns:
        --------
            np.array of float64
                Array of weight values
        '''
        return self.weights

    def predict(self, input_val):
        '''Predict a value for test based on self.weights
        Parameters:
        -----------
            input_val : np.array of float64
                Point to predict class of
        Returns:
        --------
            int (or maybe float, haven't decided yet)
                Class prediction for test
        '''
        # Concatenate bias value
        input_val = np.concatenate(([1], input_val))
        score = np.sum(input_val * self.weights)
        if score > 0:
            return 1
        return 0

    def trainSingleVal(self, input_val):
        '''Change weights based on a single point
        Parameters:
        -----------
            input_val : np.array of float64
                Point to learn on, ending with truth value
        Returns:
        --------
            None
        '''
        point = input_val[:-1]
        truth = input_val[-1]
        # Calculate weight change based on pred and point
        pred = self.predict(point)
        change_scalar = self.learning_rate * (truth - pred)
        self.weights += change_scalar * np.concatenate([[1], point])

    def trainEpoch(self):
        '''Change weights based on the training set
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        '''
        for point in self.dataset.train:
            self.trainSingleVal(point)

    def train(self, epochs=10):
        '''Change weights for a desired number of epochs
        Parameters:
        -----------
            epochs : int
                Amount of epochs to train
        Returns:
        --------
            None
        '''
        # Log initial accuracies
        self.resetAcc()
        self.logAll(0)
        for e in range(epochs):
            # Perform epoch training
            self.trainEpoch()
            if (e % 5) == 0:
                self.logAll(e)

    def evalTrain(self):
        '''
        '''
        self.resetStats()
        for point in self.dataset.train:
            p = point[:-1]
            truth = point[-1]
            self.scoreResult(self.predict(p), truth)

    def evalTest(self):
        '''
        '''
        self.resetStats()
        for point in self.dataset.test:
            p = point[:-1]
            truth = point[-1]
            self.scoreResult(self.predict(p), truth)

    def logAll(self, epoch_num):
        '''Evaluate test and train dataset partitions
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        '''
        self.evalTrain()
        self.train_acc.append(self.getBalAcc())
        self.evalTest()
        self.test_acc.append(self.getBalAcc())
        self.epoch_nums.append(epoch_num)

    def plotError(self, plot_name = 'epoch_error.png'):
        plt.figure()
        plt.plot(self.epoch_nums, np.subtract(1, self.train_acc))
        plt.plot(self.epoch_nums, np.subtract(1, self.test_acc))
        plt.legend(['Train', 'Test'])
        plt.title(f'Error vs Epoch Number @ LR = {self.learning_rate}')
        plt.xlabel('Epoch number')
        plt.ylabel('Error')
        plt.savefig(str(plot_name))
        plt.close()

    def drawDecision(
            self,
            div = None,
            plot_name = 'knn_dec_bound.png',
            plot_title = 'Decision Boundary',
            max_p = 14,
            max_n = 23,
        ):
        '''Draw decision boundary
        Parameters:
        -----------
            div : int
                Number of divisions for the axis with maximal value
                In this case, the N axis
        Returns:
        --------
            None
        '''
        p_axis_point = -self.weights[0]/self.weights[1]
        n_axis_point = -self.weights[0]/self.weights[2]
        plt.figure()
        plt.fill([0, 0, p_axis_point], [0, n_axis_point, 0], 'blue')
        plt.fill(
            [0, 0, max_p, max_p, p_axis_point],
            [n_axis_point, max_n, max_n, 0, 0],
            'orange'
        )
        plt.title(plot_title)
        plt.xlabel('P')
        plt.ylabel('N')
        plt.xlim([0, max_p])
        plt.ylim([0, max_n])
        plt.legend(['Not Stressed', 'Stressed'])
        plt.savefig(str(plot_name))
        plt.close()

if __name__ == '__main__':
    '''
    Fulfils problem 2
    '''
    # Seed RNG for repeatability
    np.random.seed(80085)
    # Additional imports
    import pathlib
    # File locations
    CODE_DIR = pathlib.Path(__file__).parent.absolute()
    ROOT_DIR = CODE_DIR.parent # Root project dir
    IMG_DIR = ROOT_DIR.joinpath('images')
    IMG_DIR.mkdir(mode=0o775, exist_ok=True)
    DATA_IN_FILE = CODE_DIR.joinpath('data.txt')
    DATA_OUT_DIR = ROOT_DIR.joinpath('data')
    DATA_OUT_DIR.mkdir(mode=0o775, exist_ok=True)
    DATA_OUT_FILE = DATA_OUT_DIR.joinpath('perceptron_err')
    ERR_PLOT = IMG_DIR.joinpath('perceptron_err.png')
    # Parameters
    LEARNING_RATE = 0.0001
    EPOCHS = 500
    # Get dataset
    dataset = Dataset(data_file = DATA_IN_FILE, shuffle_data = True)
    perc = Perceptron(dataset, learning_rate = LEARNING_RATE)
    perc.dataset.partitionXTrain(0.8)
    perc.train(epochs = EPOCHS)
    perc.plotError(ERR_PLOT)
    with open(str(DATA_OUT_FILE), 'w') as data_f:
        # Log learning rate and epochs
        data_f.write(f'learning_rate = {LEARNING_RATE}\n')
        data_f.write(f'epochs = {EPOCHS}\n')
        # Log train accuracy
        perc.evalTrain()
        train_err = 1 - perc.getBalAcc()
        data_f.write(f'train_err = {train_err:0.3f}\n')
        # Log test accuracy
        perc.evalTest()
        test_err = 1 - perc.getBalAcc()
        data_f.write(f'test_err = {test_err:0.3f}\n')

