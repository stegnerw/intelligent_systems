###############################################################################
# Imports
###############################################################################
import numpy as np
import pathlib
import matplotlib.pyplot as plt


class Dataset:
    def __init__(
            self,
            data_file = 'data.txt',
            label_file = 'labels.txt',
            train_portion = 0.8,
        ):
        '''Initialize Dataset object
        Parameters:
        -----------
            data_file : str
                Name of the file to read data from
            label_file : str
                Name of the file to read labels from
            train_portion : float
                Portion of the dataset for the train set
        Returns:
        --------
            Dataset
                The Dataset object which was constructed
        '''
        data_file = pathlib.Path(data_file).resolve()
        label_file = pathlib.Path(label_file).resolve()
        # Check that the files exist
        assert data_file.exists(), f'File not found: {str(data_file)}'
        assert label_file.exists(), f'File not found: {str(label_file)}'
        self.data_file = str(data_file)
        self.label_file = str(label_file)
        self.importDataFile()
        self.partitionData(train_portion = 0.8)

    def importDataFile(self):
        '''Imports data from the txt format Dr. Minai presented
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        '''
        # Read data from txt file
        data = list()
        with open(self.data_file) as data_f:
            data = data_f.readlines()
        data = [d.split() for d in data]
        # Read labels from txt file
        labels = list()
        with open(self.label_file) as label_f:
            labels = label_f.readlines()
        labels = [int(l) for l in labels]
        # Make images and sort into bins
        self.data_points = dict()
        self.labels = list()
        for d, l in zip(data, labels):
            if l not in self.labels:
                self.labels.append(l)
                self.data_points[l] = list()
            self.data_points[l].append(d)
        self.labels.sort()
        # Turn data lists into numpy arrays
        for l in self.labels:
            self.data_points[l] = np.array(self.data_points[l],
                    dtype=np.float64)

    def partitionData(self, train_portion=0.8):
        '''Partition the data based on train_portion
        Parameters:
        -----------
            train_portion : float, optional
                Portion of the dataset to partition to train
        Returns:
        --------
            None
        '''
        # Reset train and test partitions
        self.train_data = list()
        self.train_labels = list()
        self.test_data = list()
        self.test_labels = list()
        # Shuffle the data points first
        for l in self.labels:
            np.random.shuffle(self.data_points[l])
        # Iterate through the data and partition
        points_per_class = len(self.data_points[self.labels[0]])
        train_per_class = int(train_portion * points_per_class)
        for l in self.labels:
            for i, d in enumerate(self.data_points[l]):
                if i < train_per_class:
                    self.train_data.append(d)
                    self.train_labels.append(l)
                else:
                    self.test_data.append(d)
                    self.test_labels.append(l)
        # Turn lists into numpy arrays
        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels)
        self.test_data = np.array(self.test_data)
        self.test_labels = np.array(self.test_labels)

    def shuffleData(self, train=True, test=False):
        '''Shuffle a pair of data and labels
        Parameters:
        -----------
            train, test : bool
                Indicate which arrays should be shuffled
        Returns:
        --------
            None
        '''
        if train:
            indeces = np.random.permutation(len(self.train_data))
            self.train_data = self.train_data[indeces]
            self.train_labels = self.train_labels[indeces]
        if test:
            indeces = np.random.permutation(len(self.test_data))
            self.test_data = self.test_data[indeces]
            self.test_labels = self.test_labels[indeces]

    def getTrainBatch(self, batch_size=1):
        '''
        Parameters:
        -----------
            batch_size : int, optional
                Size of training batch
                Must divide evenly to the number of data points
        Returns:
        --------
            tuple of np.ndarray
                Batched training data of shape
                (data_points/batch_size, batch_size, 784)
                Batched training labels of shape
                (data_points/batch_size, batch_size)
        '''
        # Check batch size compatability
        assert (len(self.train_data) % batch_size) == 0, \
                'Mismatch between batch size and number of data points'
        num_batches = len(self.train_data) // batch_size
        batched_data = self.train_data.reshape(num_batches, batch_size, 784)
        batched_labels = self.train_labels.reshape(num_batches, batch_size)
        return (batched_data, batched_labels)


if __name__ == '__main__':
    np.random.seed(69420)
    from matplotlib import pyplot as plt
    import pathlib
    CODE_DIR = pathlib.Path(__file__).parent.absolute()
    ROOT_DIR = CODE_DIR.parent
    DATA_FILE = CODE_DIR.joinpath('data.txt')
    LABEL_FILE = CODE_DIR.joinpath('labels.txt')
    IMG_DIR = ROOT_DIR.joinpath('images')
    IMG_DIR.mkdir(mode=0o775, exist_ok=True)
    # Create dataset
    dataset = Dataset(DATA_FILE, LABEL_FILE)
    dataset.shuffleData(train=True, test=True)
    # Plot a sample of digits
    data, labels = dataset.getTrainBatch(100)
    fig, a = plt.subplots(10, 10)
    for i in range(100):
        img = data[1][i]
        a[i//10][i%10].imshow(img.reshape(28, 28, order='F'),
                cmap='Greys_r')
        a[i//10][i%10].axis('off')
    plt.tight_layout()
    plt.savefig('digit_samples.png')
    plt.close()
