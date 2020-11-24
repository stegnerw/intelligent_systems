###############################################################################
# Imports
###############################################################################
from settings import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pathlib
import enlighten # Progress bar for training


###############################################################################
# Classes
###############################################################################
class SOFM:
    def __init__(
            self,
            shape,
            inputs,
            weight_file = None,
            trainable = True,
            eta_0 = SOFM_ETA_0,
            eta_floor = SOFM_ETA_FLOOR,
            sigma_0 = SOFM_SIGMA_0,
            sigma_floor = SOFM_SIGMA_FLOOR,
            tau_l = SOFM_TAU_L,
            tau_n = SOFM_TAU_N,
        ):
        """Initialize Layer object either randomly or by a weight file
        Parameters
        ----------
        shape : tuple of int
            Shape of the output layer
        inputs : int
            Number of inputs
        weight_file : str, optional
            File to load pre-existing weights from
        trainable : bool, optional
            Whether or not the layer is trainable
        eta_0, eta_floor : float, optional
            Initial and final learning rate, decaying from eta_0 to eta_floor
        sigma_0, sigma_floor : float, optional
            Initial and final neighborhood radius, decaying from sigma_0 to
            sigma_floor
        tau_l, tau_n : float, optional
            Time constants (decay rates) for learning rate and neighborhood
        """
        # Assign constants
        self.trainable = trainable
        self.shape = shape
        self.num_neurons = np.prod(self.shape)
        self.inputs = inputs
        self.eta_0 = eta_0
        self.sigma_0 = sigma_0
        self.tau_l = tau_l
        self.tau_n = tau_n
        self.eta_floor = eta_floor
        self.sigma_floor = sigma_floor
        # Check if weight file was provided
        if weight_file:
            self.loadWeights(weight_file)
        else:
            self.initW()
        # States to save for back-prop
        self.e = np.empty((self.num_neurons, inputs)) # Error matrix
        self.e_norm = np.empty(self.num_neurons) # Normalized error matrix
        self.bmu = -1 # Best matching unit index
        self.d = np.empty(self.num_neurons) # Distance from BMU
        self.neighborhood = np.empty(self.num_neurons) # Neighborhood of BMU
        self.t = 0 # Current epoch number
        self.etaUpdate()
        self.sigmaUpdate()
        # Construct coordinate matrix for neighborhood distance calculation
        self.coords = np.mgrid[0:self.shape[0], 0:self.shape[1]].reshape(
                2, self.w.shape[0]).transpose()

    def initW(self):
        """Initialize weights for the perceptron uniform on [0, 1]
        """
        self.w = np.random.uniform(0, 1, size=(self.num_neurons, self.inputs))

    def forwardPass(self, x):
        """Pass the input through the network
        Parameters
        ----------
        x : np.ndarray of np.float64
            Input to the network
        """
        # Calculate and norm error
        np.subtract(x, self.w, out=self.e)
        self.e_norm = np.linalg.norm(self.e, axis=1)
        # Get best matching unit
        self.bmu = self.e_norm.argmin()

    def wUpdate(self, x):
        """Calculate weight updates for a given data point
        Parameters
        ----------
        x : np.ndarray
            1D array of an input data point
        """
        self.forwardPass(x)
        self.d = np.linalg.norm(self.coords - self.coords[self.bmu], axis=1)
        np.exp(-np.square(self.d) / (2 * self.sigma**2),
                out=self.neighborhood)
        np.add(self.eta * self.neighborhood[:, np.newaxis] * self.e, self.w,
                out=self.w)

    def etaUpdate(self):
        """Update eta based on epoch number
        """
        self.eta = max(self.eta_0 * np.exp(-self.t / self.tau_l),
                self.eta_floor)

    def sigmaUpdate(self):
        """Update sigma based on epoch number
        """
        self.sigma = max(self.sigma_0 * np.exp(-self.t / self.tau_n),
                self.sigma_floor)

    def train(self, data, epochs):
        """Train on dataset until max epochs
        Parameters
        ----------
        data : np.ndarray
            Array of training data points
        epochs : int
            Number of epochs to train
        """
        # Initialize progress bar
        pbar_manager = enlighten.get_manager()
        pbar = pbar_manager.counter(total=epochs, desc='Training',
                unit='epochs', leave=False)
        for e in range(1, 1+epochs):
            pbar.update()
            np.random.shuffle(data)
            for x in data:
                self.wUpdate(x)
            self.t += 1
            self.etaUpdate()
            self.sigmaUpdate()

    def drawFeat(self, f_name):
        """Save a grid of neuron features
        Parameters
        ----------
        f_name : pathlib.Path or str
            Name of the file to save
        """
        # New figure
        nrow = self.shape[0]
        ncol = self.shape[1]
        fig = plt.figure(figsize=((ncol+1)/2, (nrow+1)/2))
        # Configure grid
        gs = gridspec.GridSpec(nrow, ncol,
                wspace=0.1, hspace=0.1,
                top=1 - 0.75/(nrow+1), bottom=0.25/(nrow+1),
                left=0.5/(ncol), right=1 - 0.5/(ncol)
        )
        for i in range(len(self.w)):
            ax = plt.subplot(gs[self.coords[i][0], self.coords[i][1]])
            img = self.w[i].copy()
            ax.imshow(img.reshape(28, 28, order='F'), cmap='Greys_r')
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        plt.suptitle('SOFM Features', fontsize='medium')
        plt.savefig(str(f_name))
        plt.close()

    def drawHeat(self, data, labels, dir_name):
        """Save activation heat-maps of every class
        Parameters
        ----------
        data, labels : np.ndarray
            Arrays of data points and labels
        dir_name : pathlib.Path or str
            Name of the directory to save in
        """
        heatmaps = np.zeros((len(labels[0]), self.num_neurons))
        for x, l in zip(data, labels):
            c = np.argmax(l)
            self.forwardPass(x)
            heatmaps[c][self.bmu] += 1
        # Scale to probability distribution
        heatmaps /= heatmaps.sum(axis=1)[:, np.newaxis]
        heatmaps = heatmaps.reshape(len(labels[0]), self.shape[0], self.shape[1])
        for c, hm in enumerate(heatmaps):
            plt.figure()
            plt.suptitle(f'Heat Map for Class {c}')
            plt.imshow(hm, cmap='Greens')
            for i in range(len(hm)):
                for j in range(len(hm[i])):
                    color = 'k' if (hm[i][j] <= 0.5*hm.max()) else 'w'
                    plt.text(j, i, f'{hm[i][j]:0.02f}',
                        va='center', ha='center', color=color)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(str(dir_name.joinpath(f'class_{c}')),
                bbox_inches='tight', pad_inches=0)
            plt.close()

    def saveWeights(self, weight_file):
        """Save weights to a file
        Parameters
        ----------
        weight_file : str
            File name to save weights in
        """
        np.save(str(weight_file), self.w)

    def loadWeights(self, weight_file):
        """Save weights to a file
        Parameters
        ----------
        weight_file : str
            File name to save weights in
        """
        self.w = np.load(str(weight_file))


###############################################################################
# Main Code
###############################################################################
if __name__ == '__main__':
    # Additional imports
    import csv
    # Seed for consistency
    np.random.seed(SEED)
    # # Create network
    # sofm = SOFM(
        # SOFM_SHAPE,
        # INPUTS,
        # trainable = True,
        # eta_0 = SOFM_ETA_0,
        # eta_floor = SOFM_ETA_FLOOR,
        # sigma_0 = SOFM_SIGMA_0,
        # sigma_floor = SOFM_SIGMA_FLOOR,
        # tau_l = SOFM_TAU_L,
        # tau_n = SOFM_TAU_N,
    # )
    # sofm.train(train_data, SOFM_MAX_EPOCHS)

    # # Save weights and images
    # print('Saving weights...')
    # sofm.saveWeights(str(SOFM_WEIGHT_FILE))
    # print('Saving features...')
    # sofm.drawFeat(SOFM_FEAT)
    # print('Saving heat maps...')
    # sofm.drawHeat(test_data, test_labels, SOFM_HEATMAP_DIR)

    # Save parameters to CSV
    print('Saving parameters...')
    csv_rows = list()
    csv_rows.append(['Parameter', 'Value', 'Description'])
    csv_rows.append(['$out_{shape}$', f'{SOFM_SHAPE}',
        'Feature map shape'])
    csv_rows.append(['$epochs_{phase\\_1}$', f'{SOFM_PHASE_1_EPOCHS}',
        'Epochs for phase 1'])
    csv_rows.append(['$epochs_{phase\\_2}$', f'{SOFM_PHASE_2_EPOCHS}',
        'Epochs for phase 2'])
    csv_rows.append(['$\\eta_{0}$', f'{SOFM_ETA_0}',
        'Initial learning rate'])
    csv_rows.append(['$\\tau_{L}$', f'{SOFM_TAU_L:0.02f}',
        'Learning decay time constant'])
    csv_rows.append(['$\\eta_{floor}$', f'{SOFM_ETA_FLOOR}',
        'Learning rate floor for phase 2'])
    csv_rows.append(['$\\sigma_{0}$', f'{SOFM_SIGMA_0}',
        'Initial neighborhood parameter'])
    csv_rows.append(['$\\tau_{N}$', f'{SOFM_TAU_N:0.02f}',
        'Neighborhood decay time constant'])
    csv_rows.append(['$\\sigma_{floor}$', f'{SOFM_SIGMA_FLOOR}',
        'Neighborhood floor for phase 2'])
    with open(str(SOFM_PARAM_CSV), 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(csv_rows)
    print('Done.')

