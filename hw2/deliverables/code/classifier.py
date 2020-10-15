import numpy as np
import matplotlib.pyplot as plt

class Classifier():
    '''Abstract class for classifiers for Homework 2
    Used to calculate the accuracy metrics
    '''

    def __init__(self):
        '''Helps with autocomplete of parameters
        '''
        self.resetStats()

    def resetStats(self):
        '''Reset the true/false pos/neg stats
        '''
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0

    def getSens(self):
        '''Return sensitivity (recall, true positive rate)
        '''
        return float(self.true_pos) / (self.true_pos + self.false_neg)

    def getSpec(self):
        '''Return specificity (selectivity, true negative rate)
        '''
        return float(self.true_neg) / (self.false_pos + self.true_neg)

    def getPPV(self):
        '''Return positive predictive value (precision)
        '''
        return float(self.true_pos) / (self.true_pos + self.false_pos)

    def getNPV(self):
        '''Return negative predictive value
        '''
        return float(self.true_neg) / (self.true_neg + self.false_neg)

    def getAcc(self):
        '''Return accuracy (unbalanced)
        '''
        total_true = self.true_pos + self.true_neg
        total_false = self.false_pos + self.false_neg
        return float(total_true) / (total_true + total_neg)

    def getBalAcc(self):
        '''Return balanced accuracy
        '''
        return (self.getSens() + self.getSpec()) / 2.0

    def getF1(self):
        '''Return F_1 score
        '''
        return 2 / ((1/self.getPPV()) + (1/self.getSens()))

    def getFbeta(self, beta):
        '''Return F_beta score
        '''
        beta2 = beta * beta
        return (1 + beta2) / (1/(beta2 * self.getPPV()) + (1/self.getSens()))

    def scoreResult(self, pred, truth):
        '''Classify results as true/false pos/neg
        Parameters:
        -----------
            pred : int
                Predicted class
            truth : int
                Truth class
        Returns:
        --------
            None
        '''
        if truth == 0:
            if pred == 0:
                self.true_neg += 1
            else:
                self.false_pos += 1
        else:
            if pred == 0:
                self.false_neg += 1
            else:
                self.true_pos += 1

    def drawDecision(
            self,
            div = 100,
            plot_name = 'knn_dec_bound.png',
            plot_title = 'Decision Boundary',
            max_p = 14,
            max_n = 23,
        ):
        '''Draw decision boundary
        Parameters:
        -----------
            div : int
                Number of divisions for the axis with maximal value
                In this case, the N axis
        Returns:
        --------
            None
        '''
        p_step = max_p / div
        n_step = max_n / div
        p_steps = int(max_p / p_step)
        n_steps = int(max_n / n_step)
        stressed_p = list()
        stressed_n = list()
        not_stressed_p = list()
        not_stressed_n = list()
        for p_idx in range(p_steps):
            for n_idx in range(n_steps):
                p = p_idx * p_step
                n = n_idx * n_step
                # test_points[p_step][n_step][0] = p
                # test_points[p_step][n_step][1] = n
                test_point = np.array([p, n], dtype=np.float64)
                pred = self.predict(test_point)
                if pred == 0.0:
                    not_stressed_p.append(p)
                    not_stressed_n.append(n)
                if pred == 1.0:
                    stressed_p.append(p)
                    stressed_n.append(n)
        plt.figure()
        plt.scatter(not_stressed_p, not_stressed_n, c='blue', s=10,
                label='Not Stressed')
        plt.scatter(stressed_p, stressed_n, c='orange', s=10, label='Stressed')
        plt.title(plot_title)
        plt.xlabel('P')
        plt.ylabel('N')
        plt.legend(loc='upper right')
        plt.xlim([0, max_p])
        plt.ylim([0, max_n])
        plt.savefig(str(plot_name))
        plt.close()

