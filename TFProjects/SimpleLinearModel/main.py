# Add '../' into python path, so that we don't need download the data over and over again
import sys
sys.path.append('../')

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
raw_data = input_data.read_data_sets("../data/MNIST/", one_hot = True)
import TFPlots
import LinearModel

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
    linear_train_model = LinearModel.LinearModel(img_size_flat, num_classes, l_rate = 0.5, opt_type = 'GradDesc')
    linear_train_model.initialization()

    ### Show images before training
    r = 3
    c = 4
    imgs = test_set.images[0: r * c]
    cls = test_set.cls[0: r*c]
    mnist_plot.plotImage(r, c, imgs, cls, file_name = 'initial.pdf')
    print('Generate Initial Figure: DONE!')

    '''Do actuall training'''
    N_iter = 1000
    batch_size = 100
    cost_list = linear_train_model.trainingOptimize(train_set, N_iter, batch_size)
    t_list = np.arange(1,N_iter+1)
    mnist_plot.plotCost(t_list, cost_list, file_name = 'cost.pdf')
    print('Plot Cost Function: DONE!')

    '''Show the Final results'''
    cls_pred = linear_train_model.getTestPredictions(test_set)[0: r*c]
    mnist_plot.plotImage(r, c, imgs, cls, cls_pred = cls_pred, file_name = 'final.pdf')
    print('Generate Final Figure: DONE!')

    '''Plot weights'''
    wghts = linear_train_model.getWeights()
    mnist_plot.plotWeights(wghts, file_name = 'weights.pdf')
    print('Plot Weights: DONE!')



