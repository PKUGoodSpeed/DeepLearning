# Add '../' into python path, so that we don't need download the data over and over again
import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
raw_data = input_data.read_data_sets("../data/MNIST/", one_hot = True)
import TFPlots
from SimpleLinearModel import LinearModel

'''Prepare training and testing data'''
print("Size of:")
print("- Training-set:\t\t{}".format(len(raw_data.train.labels)))
print("- Test-ste:\t\t{}".format(len(raw_data.test.labels)))
print("- Validation-set:\t\t{}".format(len(raw_data.validation.labels)))

train_set = raw_data.train
test_set = raw_data.test
test_set.cls = np.argmax(test_set.labels, axis = 1)
print("Generating Training and Testing Data: DONE!")

'''Define variables for Mnist images'''
img_size = 28
img_size_flat = img_size**2
img_shape = (img_size, img_size)
num_classes = len(train_set.labels[0])

if __name__ == '__main__':
    mnist_plot = TFPlots.MnistImagePlot(height = img_size, width = img_size)
    linear_train_model = LinearModel.LinearModel(img_size_flat, num_classes, l_rate = 5000., opt_type = 'GradDesc')
    linear_train_model.initialization()

    n_iter = 40
    n_chuck = 200
    learning_rate = 8.
    n_batch = 100
    cost_list = []
    accu_list = []
    train_cost = []
    test_cost = []

    for n in range(n_iter):
        linear_train_model.modifyLearningRate(learning_rate)
        cost_list += linear_train_model.trainingOptimize(train_set, n_chuck, n_batch)
        accu_list.append(linear_train_model.getTestAccuracy(test_set))
        train_cost.append(cost_list[len(cost_list)-1])
        test_cost.append(linear_train_model.getTestCost(test_set))
        learning_rate *= 0.5

    t_list = np.arange(1,len(cost_list)+1)
    mnist_plot.plotCost(t_list, cost_list, file_name = '2/changing_lrate_cost.pdf')

    t_list = np.arange(1,len(test_cost)+1)
    mnist_plot.plotCost(t_list, test_cost, file_name='2/test_cost.pdf')
    mnist_plot.plotCost(t_list, train_cost, file_name='2/train_cost.pdf')
    mnist_plot.plotCost(t_list, accu_list, ylabel='Accuracy', file_name='2/accuracy.pdf')