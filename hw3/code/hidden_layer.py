###############################################################################
# Imports
###############################################################################
from layer import Layer
from activation import *
import numpy as np


class HiddenLayer(Layer):
    def setDownstreamSum(self, w, delta):
        '''Sum the product of w and delta for the next layer
        Needed for calculating delta for this layer
        Parameters:
        -----------
            w : np.ndarray
                Matrix of weight values for the next layer
            delta : np.ndarray
                Matrix of delta values for the next layer
        Returns:
        --------
            None
        '''
        self.downstream_sum = np.matmul(w[:, :-1].transpose(), delta)

    def setDelta(self):
        '''Calculate delta for the hidden layer
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        '''
        output_der = sigmoid_der(self.s)
        self.delta = output_der * self.downstream_sum


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

