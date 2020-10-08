###############################################################################
# Imports
###############################################################################
from dataset import Dataset
from classifier import Classifier
import numpy as np
from matplotlib import pyplot as plt


class Neighborhood(Classifier):
    def __init__(
            self,
            dataset,
            R = 1,
    ):
        self.R = R
        self.dataset = dataset
        self.resetStats()

    def setR(self, R):
        '''Set value of R
        Parameters:
        -----------
            R : int
                Value of R
        Returns:
        --------
            None
        '''
        self.R = R

    def getDistance(self, x, y):
        '''Get Euclidean distance between 2 points
        Parameters:
        -----------
            x, y : np.array of float
                Points to calculate distance between
        Returns:
        --------
            float
                Euclidean distance between x and y
        '''
        return np.sqrt(np.sum(np.square(x - y)))

    def predict(self, test):
        '''Predict a value for test based on dataset.train points
        Parameters:
        -----------
            test : np.array of float
                Point to predict class of
        Returns:
        --------
            int (or maybe float, haven't decided yet)
                Class prediction for test
        '''
        distances = list()
        for i in range(len(self.dataset.train)):
            # Store truth and distance
            new_point = dict()
            new_point['truth'] = self.dataset.train[i][2]
            new_point['dist'] = self.getDistance(test, self.dataset.train[i][:2])
            distances.append(new_point)
        # Sort distances and only keep the lowest R
        distances = [x for x in distances if x['dist'] <= self.R]
        closest_classes = {0.: 0, 1.: 0}
        for d in distances:
            closest_classes[d['truth']] += 1
        # TODO - If classes are equal it returns 0.0
        return max(closest_classes, key = lambda x: closest_classes[x])

    def evalAll(self):
        '''Predict on all data points in dataset.data_points
        For question 1
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        '''
        self.resetStats()
        # Iterate through all data points as the test points
        for i in range(len(self.dataset.data_points)):
            self.dataset.partitionOneTest(i)
            test_point = self.dataset.test[0][:2]
            test_truth = self.dataset.test[0][2]
            test_pred = self.predict(test_point)
            self.scoreResult(test_pred, test_truth)

    def evalTest(self):
        '''Predict on all data points in the test set
        For question 3
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        '''
        self.resetStats()
        # Iterate through all data points as the test points
        for p in self.dataset.test:
            test_point = p[:2]
            test_truth = p[2]
            test_pred = self.predict(test_point)
            self.scoreResult(test_pred, test_truth)


if __name__ == '__main__':
    '''
    Fulfils problem 1 part 2
    '''
    # Seed RNG for repeatability
    np.random.seed(69420)
    # Additional imports
    import pathlib
    # File locations
    CODE_DIR = pathlib.Path(__file__).parent.absolute()
    ROOT_DIR = CODE_DIR.parent # Root project dir
    IMG_DIR = ROOT_DIR.joinpath('images')
    IMG_DIR.mkdir(mode=0o775, exist_ok=True) # Create images dir if needed
    BAL_ACC_PLOT = IMG_DIR.joinpath('neighborhood_bal_acc.png')
    DATA_IN_FILE = CODE_DIR.joinpath('data.txt')
    DATA_OUT_DIR = ROOT_DIR.joinpath('data')
    DATA_OUT_DIR.mkdir(mode=0o775, exist_ok=True)
    DATA_OUT_FILE = DATA_OUT_DIR.joinpath('neighborhood_acc')
    # Get dataset
    dataset = Dataset(data_file = DATA_IN_FILE, shuffle_data = True)
    neighborhood = Neighborhood(dataset, R=1)
    # Iterate through R values
    R_vals = np.arange(0, 10, 0.2)
    bal_acc = list()
    for R in R_vals:
        neighborhood.setR(R)
        neighborhood.evalAll()
        bal_acc.append(neighborhood.getBalAcc())
        print(f'R = {R:0.1f}\tBalanced accuracy: {neighborhood.getBalAcc()}')
    max_r = R_vals[np.argmax(bal_acc)]
    print(f'Maximum accuracy: {np.max(bal_acc)}\tr = {max_r:0.1f}')
    # Plot data points
    plt.plot(R_vals, bal_acc)
    plt.title('Neighborhood Classifier Balanced Accuracy vs R')
    plt.xlabel('R')
    plt.ylabel('Balanced Accuracy')
    plt.savefig(str(BAL_ACC_PLOT))
    # Get accuracy at optimal R
    neighborhood.setR(max_r)
    neighborhood.evalAll()
    with open(str(DATA_OUT_FILE), 'w') as data_f:
        data_f.write(f'bestr = {max_r:0.1f}\n')
        neighborhood.evalAll()
        acc = 100 * neighborhood.getBalAcc()
        data_f.write(f'acc = {acc:0.2f}\n')

