# Add '../' into python path, so that we don't need download the data over and over again
import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
raw_data = input_data.read_data_sets("../data/MNIST/", one_hot = True)
import TFPlots
import TwoConvCNN

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
img_channel = 1
num_classes = len(train_set.labels[0])

filter_size1 = 5
num_filter1 = 16
filter_size2 = 5
num_filter2 = 36
num_fc1 = 128
num_fc2 = num_classes

conv_shape1 = [filter_size1, filter_size1, img_channel, num_filter1]
conv_shape2 = [filter_size2, filter_size2, num_filter1, num_filter2]
fc_shape1 = [-1, num_fc1]
fc_shape2 = [num_fc1, num_fc2]

def selectLearningRate(lower, upper, ratio, N_step):
    '''
    The method to check Learning rate
    :param lower: lower_bound
    :param upper: upper_bound
    :param ratio: retio
    :param N_step: number of steps for the iteration
    :return: None
    '''
    mnist_plot = TFPlots.MnistImagePlot(height=img_size, width=img_size)
    CNN_model = TwoConvCNN.TwoConvCNN([img_size, img_size, img_channel], num_classes)
    CNN_model.constructModel(conv_shape1, conv_shape2, fc_shape1, fc_shape2)

    l_rate = lower
    l_rate_list = []
    accu_list = []
    while l_rate < upper:
        CNN_model.initialization(l_rate, opt_type = 'Adam')
        CNN_model.training(train_set, N_step, 64)
        l_rate_list.append(l_rate)
        accuracy = CNN_model.getTestAccuracy(test_set)
        accu_list.append(accuracy)
    mnist_plot.plotCost(l_rate_list, accu_list, xscale = 'log', xlabel = 'Learning rate',
                        ylabel = 'Temp Accuracy', file_name = 'lrate_vs_accu.pdf')
    return l_rate_list, accu_list


if __name__ == '__main__':
    lower = 1.e-6
    upper = 1.e-1
    ratio = 2.
    sele_number = 1000
    l_rate_list, accu_list = selectLearningRate(lower, upper, ratio, sele_number)
    for i, l_rate in enumerate(l_rate_list):
        print(i, "{0}, Training Accuracy: {1:>6.1%}".format(l_rate, accu_list[i]))

    mnist_plot = TFPlots.MnistImagePlot(height=img_size, width=img_size)
    CNN_model = TwoConvCNN.TwoConvCNN([img_size, img_size, img_channel], num_classes)
    CNN_model.constructModel(conv_shape1, conv_shape2, fc_shape1, fc_shape2)








    """
    CNN_model.initialization(l_rate = 1.e-4, opt_type = 'Adam')

    ### Show images before training
    r = 3
    c = 4
    imgs = test_set.images[0: r * c]
    cls = test_set.cls[0: r*c]
    mnist_plot.plotImage(r, c, imgs, cls, file_name = 'initial.pdf')
    print('Generate Initial Figure: DONE!')

    CNN_model.training(train_set, num_iterations = 51, batch_size = 64)
    t_list, accu_list, cost_list = CNN_model.getTrainingResults()
    mnist_plot.plotCost(t_list, accu_list, ylabel = 'accuracy')
    mnist_plot.plotCost(t_list, cost_list)

    accuracy, cls_pred = CNN_model.getTestAccuracy(test_set)
    mnist_plot.plotImage(r, c, imgs, cls, cls_pred = cls_pred[0: r*c])
    print("Generate Final Figure: DONE")
    """
