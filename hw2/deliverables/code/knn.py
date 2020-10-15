###############################################################################
# Imports
###############################################################################
from dataset import Dataset
from classifier import Classifier
import numpy as np
from matplotlib import pyplot as plt


class KNN(Classifier):
    def __init__(
            self,
            dataset,
            k = 11,
    ):
        self.k = k
        self.dataset = dataset
        self.resetStats()

    def setK(self, k):
        '''Set value of k
        Parameters:
        -----------
            k : int
                Value of k
        Returns:
        --------
            None
        '''
        self.k = k

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
        # What a one-line beauty
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
        # Sort distances and only keep the lowest k
        distances.sort(key=lambda x: x['dist'])
        distances = distances[:self.k]
        closest_classes = {0.: 0, 1.: 0}
        for d in distances:
            closest_classes[d['truth']] += 1
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
    Fulfils problem 1 part 1
    '''
    # Seed RNG for repeatability
    np.random.seed(69420)
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
    DATA_OUT_FILE = DATA_OUT_DIR.joinpath('knn_acc')
    BAL_ACC_PLOT = IMG_DIR.joinpath('knn_bal_acc.png')
    # Get dataset
    dataset = Dataset(data_file = DATA_IN_FILE, shuffle_data = True)
    knn = KNN(dataset, k=1)
    # Iterate through k values
    k_vals = range(1, 100, 2)
    bal_acc = list()
    for k in k_vals:
        knn.setK(k)
        knn.evalAll()
        bal_acc.append(knn.getBalAcc())
        print(f'k = {k}\tBalanced accuracy: {knn.getBalAcc()}')
    max_k = k_vals[np.argmax(bal_acc)]
    print(f'Maximum accuracy: {np.max(bal_acc)}\tk = {max_k}')
    # Plot data points
    plt.figure()
    plt.plot(k_vals, bal_acc)
    plt.title('k-Nearest Neighbors Balanced Accuracy vs k')
    plt.xlabel('k')
    plt.ylabel('Balanced Accuracy')
    plt.savefig(str(BAL_ACC_PLOT))
    plt.close()
    # Get accuracy at optimal k
    knn.setK(max_k)
    knn.evalAll()
    with open(str(DATA_OUT_FILE), 'w') as data_f:
        data_f.write(f'bestk = {max_k}\n')
        knn.evalAll()
        acc = 100 * knn.getBalAcc()
        data_f.write(f'acc = {acc:0.2f}\n')

