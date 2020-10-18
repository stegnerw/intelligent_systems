###############################################################################
# Imports
###############################################################################
from layer import Layer
from activation import *
import numpy as np


class OutputLayer(Layer):
    def setLabel(self, label):
        '''Set the labels for this batch
        Needed for calculating delta for this layer
        Parameters:
        -----------
            label : np.ndarray
                One-hot encoded label
        Returns:
        --------
            None
        '''
        self.label = label

    def setDelta(self):
        '''Calculate delta for the output layer
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        '''
        e = self.label - self.y
        y_der = sigmoid_der(self.s)
        self.delta = e * y_der

    def thresholdOutputs(self, L, H):
        '''Threshold outputs
        Parameters:
        -----------
            L, H: float
                Low and high thresholds for outputs
        Returns:
        --------
            None
        '''
        self.y[self.y >= H] = 1.0
        self.y[self.y <= L] = 0.0


if __name__ == '__main__':
    # Seed for consistency
    np.random.seed(69420)
    # Additional imports
    from dataset import Dataset
    import pathlib
    # File locations
    CODE_DIR = pathlib.Path(__file__).parent.absolute()
    ROOT_DIR = CODE_DIR.parent
    DATA_FILE = CODE_DIR.joinpath('data.txt')
    LABEL_FILE = CODE_DIR.joinpath('labels.txt')
    IMG_DIR = ROOT_DIR.joinpath('images')
    IMG_DIR.mkdir(mode=0o775, exist_ok=True)

