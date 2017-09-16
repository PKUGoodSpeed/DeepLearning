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
    
    '''
    test_model = MultiConvCNN.CheckingParameters(img_shape, num_classes, conv_shape_list, conv_pool_list,
                                                 conv_relu_list, fc_shape_list, fc_relu_list)
    lower = 3.e-4
    upper = 1.e-2
    ratio = 1.25
    n_step = 1001
    lrate_list, accu_list = test_model.checkLearningRate(train_set, test_set, lower, upper, ratio, n_step)
    mnist_plot.plotCost(lrate_list, accu_list, xscale = 'log', xlabel = 'Learning rate',
                        ylabel = 'Test accuracy', file_name = 'main2_results/lr_vs_accu2.pdf')
    Optimal learning rate ~ 1.e-3
    '''
    
    running_model = MultiConvCNN.MultiConvLayerCNN(img_shape, num_classes)
    running_model.constuctCNNetwork(conv_shape_list, conv_pool_list, conv_relu_list, fc_shape_list, fc_relu_list)
    running_model.initialization(l_rate = 1.e-3, opt_type = 'Adam')
    running_model.trainData(train_set, 10000, batch_size = 64, output_step = 50)
    step_list, accu_list, cost_list = running_model.getTrainingResults()
    
    # Graphically showing data
    print("\n\n============Plotting Data============\n\n")
    mnist_plot.plotCost(step_list , accu_list, ylabel = 'Accuracy', file_name = 'main2_results/accuracy.pdf')
    mnist_plot.plotCost(step_list , cost_list, ylabel = 'Cost', file_name = 'main2_results/cost.pdf')
    
    r = 3
    c = 4
    imgs = test_set.images[: r*c]
    cls = test_set.cls[: r*c]
    test_accu, cls_pred = running_model.getTestAccuracy(test_set, n_cls_pred = r*c)
    print("The Final Accuracy is {0>6.2%}".format(test_accu))
    mnist_plot.plotImage(r, c, imgs, cls, cls_pred = cls_pred, file_name = 'final_prediction_example.pdf')
