###############################################################################
# Imports
###############################################################################
from layer import Layer
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
        y_der = self.y * (1 - self.y)
        self.delta = e * y_der

    def thresholdOutputs(self, L, H):
        '''Threshold outputs based on the labels
        Parameters:
        -----------
            L, H: float
                Low and high thresholds for outputs
        Returns:
        --------
            None
        '''
        self.y[(self.y <= L) * (self.label == 0)] = 0
        self.y[(self.y >= H) * (self.label == 1)] = 1


if __name__ == '__main__':
    print('Warning: Tests for this file are deprecated')

