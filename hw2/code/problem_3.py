###############################################################################
# Imports
###############################################################################
from dataset import Dataset
from knn import KNN
from neighborhood import Neighborhood
from perceptron import Perceptron
import numpy as np
from matplotlib import pyplot as plt
import pathlib


###############################################################################
# Best Metrics / Constants
###############################################################################
K = 11
R = 2.8
LEARNING_RATE = 0.0001
EPOCHS = 150
# File locations
CODE_DIR = pathlib.Path(__file__).parent.absolute()
ROOT_DIR = CODE_DIR.parent # Root project dir
IMG_DIR = ROOT_DIR.joinpath('images')
IMG_DIR.mkdir(mode=0o775, exist_ok=True)
KNN_PERF = IMG_DIR.joinpath('knn_performance.png')
KNN_DEC_BOUND = IMG_DIR.joinpath('knn_dec_bound.png')
NEIGHBOR_DEC_BOUND = IMG_DIR.joinpath('neighborhood_dec_bound.png')
NEIGHBOR_PERF = IMG_DIR.joinpath('neighborhood_performance.png')
PERC_DEC_BOUND = IMG_DIR.joinpath('perceptron_dec_bound.png')
PERC_PERF = IMG_DIR.joinpath('perceptron_performance.png')
TRIAL_ERR = IMG_DIR.joinpath('trial_wise_error.png')
MEAN_ERR = IMG_DIR.joinpath('mean_error.png')
DATA_IN_FILE = CODE_DIR.joinpath('data.txt')
DATA_OUT_DIR = ROOT_DIR.joinpath('data')
DATA_OUT_DIR.mkdir(mode=0o775, exist_ok=True)
AVG_PERF_TAB = DATA_OUT_DIR.joinpath('avg_perf.csv')


###############################################################################
# Problem 3
###############################################################################
# Function definitions
def logMetrics(classifier, key):
    '''Log performance metrics for the given classifier
    Parameters:
    -----------
        classifier : Classifier
            Classifier object to get metrics from
            Should be KNN, Neighborhood, or Perceptron
        key : str
            Key for list to store metrics in
    '''
    bal_acc[key].append(classifier.getBalAcc())
    precision[key].append(classifier.getPPV())
    recall[key].append(classifier.getSens())
    f1[key].append(classifier.getF1())

def bestDecision(classifier, key, plot_name, plot_title):
    if classifier.getBalAcc() > maxAcc[key]:
        print('* New best accuracy')
        maxAcc[key] = classifier.getBalAcc()
        classifier.drawDecision(plot_name=plot_name,
                plot_title=plot_title)

def plotPerf(keys, plot_name, plot_title = '', legend = None):
    bar_width = .4
    x = np.arange(1, 10)
    plt.figure()
    plt.suptitle(plot_title)
    # Balanced Accuracy
    plt.subplot(221)
    plt.xlabel('Iteration')
    plt.ylabel('Balanced Accuracy')
    plt.ylim([0, 1])
    for i, key in enumerate(keys):
        plt.bar((i * bar_width) + x, bal_acc[key], width = bar_width)
    # Precision
    plt.subplot(222)
    plt.xlabel('Iteration')
    plt.ylabel('Precision')
    plt.ylim([0, 1])
    for i, key in enumerate(keys):
        plt.bar((i * bar_width) + x, precision[key], width = bar_width)
    # Recall
    plt.subplot(223)
    plt.xlabel('Iteration')
    plt.ylabel('Recall')
    plt.ylim([0, 1])
    for i, key in enumerate(keys):
        plt.bar((i * bar_width) + x, recall[key], width = bar_width)
    # F1
    plt.subplot(224)
    plt.xlabel('Iteration')
    plt.ylabel('F1 Score')
    plt.ylim([0, 1])
    for i, key in enumerate(keys):
        plt.bar((i * bar_width) + x, f1[key], width = bar_width)
    if legend != None:
        for idx in range(221, 225):
            plt.subplot(idx)
            plt.legend(legend)
    plt.tight_layout()
    plt.savefig(str(plot_name))
    plt.close()

# Seed RNG for repeatability
np.random.seed(69420)
# Store results for later graphing
bal_acc = {
    'knn': list(),
    'neighborhood': list(),
    'perc_train': list(),
    'perc_test': list(),
}
precision = {
    'knn': list(),
    'neighborhood': list(),
    'perc_train': list(),
    'perc_test': list(),
}
recall = {
    'knn': list(),
    'neighborhood': list(),
    'perc_train': list(),
    'perc_test': list(),
}
f1 = {
    'knn': list(),
    'neighborhood': list(),
    'perc_train': list(),
    'perc_test': list(),
}
train_err = list()
maxAcc = {
    'knn': 0,
    'neighborhood': 0,
    'perceptron': 0,
}
# Test 9 versions of the dataset/algorithms
for i in range(9):
    print(f'* * * Starting iteration {i+1} of 9 * * *')
    dataset = Dataset(data_file=DATA_IN_FILE, shuffle_data=True)
    dataset.partitionXTrain(0.8)
    print(f'Starting KNN')
    knn = KNN(dataset, k=K)
    knn.evalTest()
    logMetrics(knn, 'knn')
    bestDecision(knn, 'knn', KNN_DEC_BOUND,
            f'k-Nearest Neighbors Decision Boundary @ k={knn.k}')
    print(f'Starting Neighborhood')
    neighborhood = Neighborhood(dataset, R=R)
    neighborhood.evalTest()
    logMetrics(neighborhood, 'neighborhood')
    bestDecision(neighborhood, 'neighborhood', NEIGHBOR_DEC_BOUND,
            f'Neighborhood Decision Boundary @ R={neighborhood.R}')
    print(f'Starting Perceptron')
    perceptron = Perceptron(dataset, learning_rate=LEARNING_RATE)
    perceptron.train(epochs=EPOCHS)
    perceptron.evalTrain()
    logMetrics(perceptron, 'perc_train')
    perceptron.evalTest()
    logMetrics(perceptron, 'perc_test')
    train_err.append(np.subtract(1, perceptron.train_acc))
    bestDecision(perceptron, 'perceptron', PERC_DEC_BOUND,
            f'Perceptron Decision Boundary')
# Individual trial performance
plotPerf(['knn'], KNN_PERF, 'KNN Performance')
plotPerf(['neighborhood'], NEIGHBOR_PERF, 'Neighborhood Performance')
plotPerf(['perc_train', 'perc_test'], PERC_PERF, 'Perceptron Performance',
        legend=['Train', 'Test'])
# Average performance
metrics = [bal_acc, precision, recall, f1]
labels = ['Balanced Accuracy', 'Precision', 'Recall', 'F1 Score']
keys = ['knn', 'neighborhood', 'perc_test']
with open(str(AVG_PERF_TAB), 'w') as csv:
    csv.write(',KNN,Neighborhood,Perceptron\n')
    for metric, metric_l in zip(metrics, labels):
        write_str = f'{metric_l},'
        for i in range(len(keys)):
            mean = np.mean(metric[keys[i]])
            std = np.std(metric[keys[i]])
            write_str += f'${mean:0.3f} \pm {std:0.3f}$'
            if i == (len(keys) - 1):
                write_str += '\n'
            else:
                write_str += ','
        csv.write(write_str)
# Trial-wise training error
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Trial-Wise Training Error vs Epochs')
for err in train_err:
    plt.plot(perceptron.epoch_nums, err)
plt.legend([f'Trial {i}' for i in range(1, 10)], loc='upper right')
plt.savefig(str(TRIAL_ERR))
plt.close()
# Mean training error
mean_err = []
std_err = []
for i in range(len(train_err[0])):
    points = []
    for j in range(len(train_err)):
        points.append(train_err[j][i])
    mean_err.append(np.mean(points))
    std_err.append(np.std(points))
plt.figure()
plt.errorbar(perceptron.epoch_nums, mean_err, yerr=std_err)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Mean Training Error vs Epochs')
plt.savefig(str(MEAN_ERR))
plt.close()

