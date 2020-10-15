###############################################################################
# Imports
###############################################################################
import numpy as np


class Dataset:
    def __init__(
            self,
            data_file = 'data.txt',
            classes = ['Not Stressed', 'Stressed'],
            shuffle_data = False,
    ):
        self.data_file = str(data_file)
        self.classes = classes
        self.importDataFile()
        if shuffle_data:
            self.shuffleData()

    def isFloat(self, test_str):
        '''Quick check if a string is a float value
        Parameters:
        -----------
            test_str : str
                String to test for float cast compatibility
        Returns:
            bool
                True if the string can cast to a float, else false
        '''
        try:
            float(test_str)
            return True
        except ValueError:
            return False

    def importDataFile(self):
        '''Imports data from the txt format Dr. Minai presented
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        '''
        print(f'Importing text data from {self.data_file}')
        self.data_points = list()
        # Read data from txt file
        with open(self.data_file) as data:
            class_id = 0
            for line in data.readlines():
                line = line[:-1] # Strip off newline
                # Check if line is new class definition
                if line in self.classes:
                    class_id = self.classes.index(line)
                    continue
                # Check if line is a set of data points
                line = line.split()
                if (len(line) == 2) and self.isFloat(line[0]) and self.isFloat(line[1]):
                    # Each data point will be [P, N, Truth]
                    new_item = np.array(
                        (float(line[0]), float(line[1]), float(class_id)),
                        dtype=np.float64
                    )
                    self.data_points.append(new_item)
        num_data_points = len(self.data_points)
        self.data_points = np.array(self.data_points, dtype=np.float64)

    def resizeDataPartitions(self, num_train_points):
        '''Size dataset partitions
        Parameters:
        -----------
            num_train_points : int
                Number of train data points
                The rest will be test points
        Returns:
        --------
            None
        '''
        num_test_points = len(self.data_points) - num_train_points
        self.train = np.empty((num_train_points, 3), dtype=np.float64)
        self.test = np.empty((num_test_points, 3), dtype=np.float64)

    def partitionAllTrain(self):
        '''Format entire dataset to training
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        '''
        # Allocate train and test numpy arrays
        num_train_points = len(self.data_points)
        self.resizeDataPartitions(num_train_points)
        # Iterate through data points
        for i in range(len(self.data_points)):
            self.train[i] = self.data_points[i]

    def partitionOneTest(self, skip_idx):
        '''Format all but one data point to training
        Parameters:
        -----------
            skip_idx : int
                Index of testing data point
        Returns:
        --------
            None
        '''
        num_train_points = len(self.data_points) - 1
        self.resizeDataPartitions(num_train_points)
        # Iterate through data points except for skip_idx
        for i in range(len(self.data_points)):
            if i == skip_idx:
                self.test[0] = self.data_points[i]
                continue
            item_idx = i
            if i > skip_idx:
                item_idx -= 1
            self.train[item_idx] = self.data_points[i]

    def partitionXTrain(self, train_portion):
        '''Format a portion of the dataset to training
        Parameters:
        -----------
            train_portion : float
                Portion of data to make train
        Returns:
        --------
            None
        '''
        num_train_points = int(train_portion * len(self.data_points))
        self.resizeDataPartitions(num_train_points)
        # Iterate through data_points and partition
        for i in range(len(self.data_points)):
            if i < num_train_points:
                self.train[i] = self.data_points[i]
            else:
                item_idx = i - num_train_points
                self.test[item_idx] = self.data_points[i]

    def shuffleData(self):
        '''Shuffle data in self.data_points
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        '''
        np.random.shuffle(self.data_points)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import pathlib
    CODE_DIR = pathlib.Path(__file__).parent.absolute()
    ROOT_DIR = CODE_DIR.parent
    DATA_FILE = CODE_DIR.joinpath('data.txt')
    IMG_DIR = ROOT_DIR.joinpath('images')
    IMG_DIR.mkdir(mode=0o775, exist_ok=True)
    # Plot data points
    dataset = Dataset(data_file = DATA_FILE)
    dataset.partitionAllTrain()
    not_stressed = {'p': list(), 'n': list()}
    stressed = {'p': list(), 'n': list()}
    for point in dataset.train:
        if point[2] == 0: # Not stressed
            not_stressed['p'].append(point[0])
            not_stressed['n'].append(point[1])
        else: # Stressed
            stressed['p'].append(point[0])
            stressed['n'].append(point[1])
    plt.scatter(not_stressed['p'], not_stressed['n'], label='Not Stressed')
    plt.scatter(stressed['p'], stressed['n'], label='Stressed')
    plt.xlabel('P')
    plt.ylabel('N')
    plt.legend()
    data_plot = IMG_DIR.joinpath('dataset.png')
    plt.savefig(str(data_plot))

