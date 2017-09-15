# Add '../' into python path, so that we don't need download the data over and over again
import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
raw_data = input_data.read_data_sets("../data/MNIST/", one_hot = True)
import TFPlots
import MultiConvCNN

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
img_channel = 1
img_shape = [img_size, img_size, img_channel]
num_classes = len(train_set.labels[0])

fltr_shape1 = [5, 5, 1, 16]
fltr_shape2 = [5, 5, 16, 36]
fltr_shape3 = [5, 5, 36, 17]
conv_shape_list = [fltr_shape1, fltr_shape2, fltr_shape3]
conv_pool_list = [True, True, False]
conv_relu_list = [True, True, True]

fc_shape1 = [0, 128]
fc_shape2 = [128, num_classes]
fc_shape_list = [fc_shape1, fc_shape2]
fc_relu_list = [True, True]

if __name__ == '__main__':
    mnist_plot = TFPlots.MnistImagePlot(height = img_size, width = img_size)
    test_model = MultiConvCNN.CheckingParameters(img_shape, num_classes, conv_shape_list, conv_pool_list,
                                                 conv_relu_list, fc_shape_list, fc_relu_list)
    lower = 1.e-5
    upper = 2.e-3
    ratio = 2.
    n_step = 201
    lrate_list, accu_list = test_model.checkLearningRate(train_set, test_set, lower, upper, ratio, n_step)
    mnist_plot.plotCost(lrate_list, accu_list, xscale = 'log', xlabel = 'Learning rate',
                        ylabel = 'Test accuracy', file_name = 'main2_results/lr_vs_accu1.pdf')
