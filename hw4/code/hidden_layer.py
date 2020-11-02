###############################################################################
# Imports
###############################################################################
from layer import Layer
import numpy as np


class HiddenLayer(Layer):
    def setDownstreamSum(self, w, delta):
        """Sum the product of w and delta for the next layer
        Needed for calculating delta for this layer
        Parameters
        ----------
        w : np.ndarray
            Matrix of weight values for the next layer
        delta : np.ndarray
            Matrix of delta values for the next layer
        """
        self.downstream_sum = np.matmul(w[:,:-1].transpose(), delta)

    def setDelta(self):
        """Calculate delta for the hidden layer
        """
        # Derivative of sigmoid using last forward pass
        output_der = self.y * (1 - self.y)
        self.delta = output_der * self.downstream_sum


if __name__ == '__main__':
    print('Warning: Tests for this file are deprecated')

