###############################################################################
# Imports
###############################################################################
from layer import Layer
from activation import *
import numpy as np


class OutputLayer(Layer):
    def setLabels(self, labels):
        '''Set the labels for this batch
        Needed for calculating delta for this layer
        Parameters:
        -----------
            labels : np.ndarray
                Matrix of labels for the current batch
        Returns:
        --------
            None
        '''
        self.labels = labels

    def setDelta(self):
        '''Calculate delta for the output layer
        Parameters:
        -----------
            None
        Returns:
        --------
            None
        '''
        error = self.labels - self.output
        output_der = sigmoid_der(self.net_inputs)
        print(f'error: {error.shape}\noutput_der: {output_der.shape}')
        self.delta = error * output_der


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

