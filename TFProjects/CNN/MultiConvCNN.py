'''
Here I try to implement a Model in which we can specify the number of layers
'''
import numpy as np
import tensorflow as tf
import time
from datetime import timedelta

class MultiConvLayerCNN:
    '''
    We are constructing a model in which we can manually specify
    the number of convolutional/fully connected and their dimensions
    Current code is only for Mnist-Image Classification
    '''

    __optimizer_list = {
        'GradDesc': tf.train.GradientDescentOptimizer,
        'Adam': tf.train.AdamOptimizer,
        'AdGrad': tf.train.AdagradOptimizer,
    }
    __img_shape = None
    __num_cls = None
    __x = None
    __x_img = None
    __y = None
    __y_cls = None
    __y_pred = None
    __y_cls_pred = None

    def __init__(self, img_shape, num_cls):
        '''
        Initialization of the class
        :param img_shape: image shape = [ image height, image width, image channels ], type == list of numbers
        :param num_cls: number of classes
        '''
        ## First we define all the place holder variables
        self.__img_shape = img_shape
        self.__num_cls = num_cls
        self.__x = tf.placeholder(tf.float32, shape = [None, np.array(img_shape).prod()])
        self.__x_img = tf.reshape(self.__x, shape = [-1] + list(img_shape))
        self.__y = tf.placeholder(tf.float32, shape = [None, num_cls])
        self.__y_cls = tf.argmax(self.__y, axis = 1)

    def generateNewWeights(self, weights_shape, dev = 0.05):
        '''
        Generating new weights
        :param weights_shape: dimensions of the weight, type == list of numbers
        :return: new weights
        '''
        return tf.Variable(tf.truncated_normal(shape = weights_shape, stddev = dev))

    def generateNewBiases(self, biases_shape, value = 0.05):
        '''
        Generating new biases
        :param biases_shape: dimensions of the biases, type == list
        :return: new biases
        '''
        return tf.Variable(tf.constant( value, shape = biases_shape))

    def createConvLayer(self, input, layer_shape, pooling = True, relu = True):
        '''
        Creating a convolutional layer
        :param input: last conv layer or the input images
        :param layer_shape: list( [filter height, filter width, filter depth, number of filters] )
        :param pooling: whether do down sampling
        :param relu: whether use relu activation
        :return: new conv layer
        '''
        # Generate a set of weights and biases
        weights = self.generateNewWeights(layer_shape)
        biases = self.generateNewBiases(layer_shape[3:])

        # Create a new layer
        layer = tf.nn.conv2d(input = input, strides = [1, 1, 1, 1], filter = weights, padding = 'SAME')
        layer += biases

        # Whether doing relu
        if relu:
            layer = tf.nn.relu(layer)

        # Whether doing down sampling or pooling
        if pooling:
            layer = tf.nn.max_pool(layer, strides = [1, 2, 2, 1], ksize = [1, 2, 2, 1], padding = 'SAME')

        # Output messages and return layer
        print("Creating a New Convolutional Layer:")
        print("Filter size: {0} X {1} X {2}".format(*layer_shape[:3]))
        print("Number of filters: {}".format(layer_shape[3]))
        print("------------DONE!------------\n\n")
        return layer, weights, biases

    def flattenLayer(self, input_layer):
        '''
        Flatten the last conv layer
        :param input_layer: the last conv_layer
        :return: a flat layer
        '''
        # Get the dimensions (aborting the first dimension)
        layer_shape = input_layer.get_shape().as_list()
        num_features = np.array(layer_shape[1: ], dtype = np.int64).prod()

        # Creating a flatten layer
        flat_layer = tf.reshape(input_layer, shape = [-1, num_features])

        # Output messages and reture layer
        print("Flatten a Layer:")
        print("Convert a {0} X {1} X {2} Convolutional layer".format(*layer_shape[1:]))
        print("into a flat layer with {} neuron(s)".format(num_features))
        print("------------DONE!------------\n\n")
        return flat_layer, num_features

    def createFCLayer(self, input, layer_shape, relu = True):
        '''
        Creating a fully-connected layer
        :param input: the last FC layer or a flat layer
        :param layer_shape: list( [input size, output size] )
        :param relu: whether use relu activation
        :return: a FC layer
        '''
        # Generate a set of weights and biases
        weights = self.generateNewWeights(layer_shape)
        biases = self.generateNewBiases(layer_shape[1:])

        # Creating a new layer
        layer = tf.matmul(input, weights) + biases

        # Output messages and reture layer
        print("Creating a New Fully-connected Layer:")
        print("Input size: {}".format(layer_shape[0]))
        print("Output size: {}".format(layer_shape[1]))
        print("------------DONE!------------\n\n")
        return layer, weights, biases

    def constuctCNNetwork(self, conv_shape_list, conv_pool_list, conv_relu_list, fc_shape_list, fc_relu_list):
        '''
        Construct the Convolutional Neural Network, which consist of several conv layers + a flat layer
        + several FC layers
        :param conv_shape_list: list of conv layer shapes, type == list of lists
        :param conv_pool_list: list of pooling controls for conv layers, type == list of booleans
        :param conv_relu_list: list of activation control, type == list of booleans
        :param fc_shape_list: list of FC layer shapes, type == list of lists
        :param fc_relu_list: list of pooling controls for FC layers, type == list of booleans
        :return: None
        '''
        n_conv = len(conv_shape_list)
        n_fc = len(fc_shape_list)
        assert n_conv > 0 and n_fc > 0
        assert len(conv_pool_list) == len(conv_relu_list) == n_conv
        assert len(fc_relu_list) == n_fc
        assert conv_shape_list[0][2] == self.__img_shape[2]

        # Generate the first Convolutional layer
        conv_layer, _, _ = self.createConvLayer(self.__x_img, conv_shape_list[0],
                                          pooling = conv_pool_list[0], relu = conv_relu_list[0])

        # Generate a series of Convolutional layers
        for i in range(1, n_conv):
            assert conv_shape_list[i][2] == conv_shape_list[i-1][3]
            conv_layer, _, _ = self.createConvLayer(conv_layer, conv_shape_list[i],
                                              pooling = conv_pool_list[i], relu = conv_relu_list[i])

        # Flatten the last Convolutional layer
        flat_layer, num_features = self.flattenLayer(conv_layer)

        # Generate the first fully-connected layer
        fc_shape_list[0][0] = num_features
        fc_layer, _, _ = self.createFCLayer(flat_layer, fc_shape_list[0], relu = fc_relu_list[0])

        # Generate a series of fully-connected layers
        for i in range(1, n_fc):
            assert fc_shape_list[i][0] == fc_shape_list[i-1][1]
            fc_layer, _, _ = self.createFCLayer(fc_layer, fc_shape_list[i])

        # Define output variables
        self.last_layer = fc_layer
        assert fc_shape_list[n_fc-1][1] == self.__num_cls
        self.__y_pred = tf.nn.softmax(self.last_layer)
        self.__y_cls_pred = tf.argmax(self.__y_pred, axis = 1)

        # Output Messages and return None
        print("Construct the Convolutional Neural Network:")
        print("{0} X Conv Layer(s) + 1 X Flat Layer + {1} X FC Layer(s)".format(n_conv, n_fc))
        print("============DONE!============\n\n")
        return None

    def initialization(self, l_rate, opt_type = 'GradDesc'):
        '''
        Defining the target variables, cost/loss function and optimizer
        Starting a Tensorflow session
        :param l_rate: learning rate
        :param opt_type: optimizer type
        :return: None
        '''
        # Define Entropy or Lost Function or Cost function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.last_layer, labels=self.__y)
        self.cost = tf.reduce_mean(cross_entropy)

        # Define Optimizer and Optimization variables
        assert opt_type in self.__optimizer_list.keys()
        self.optimizer = self.__optimizer_list[opt_type](learning_rate=l_rate).minimize(self.cost)

        # Define Performance measurement
        correct_predictions = tf.equal(self.__y_cls_pred, self.__y_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        # Start a session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        # Initialize some parameters
        self.total_iteration = 0
        self.accu_list = []
        self.step_list = []
        self.cost_list = []

        # Show status
        print("Ready for training!\n\n")
        return None

    def trainData(self, train_data, num_iterations, batch_size, output_step = 50):
        '''
        Train the training set for certain number of iterations
        :param train_data: training examples
        :param num_iterations: number of iterations
        :param batch_size: batch size
        :param output_step: Every output_step number of iterations, the program will output an accuracy value
        :return: None
        '''
        start_time = time.time()
        for i in range(num_iterations):

            # Get training set
            x_batch, y_batch = train_data.next_batch(batch_size)
            train_set = {
                self.__x: x_batch,
                self.__y: y_batch,
            }
            self.session.run(self.optimizer, feed_dict=train_set)
            if (self.total_iteration + i) % output_step == 0:
                accu, cost = self.session.run([self.accuracy, self.cost], feed_dict=train_set)

                # Message for printing
                msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.2%}"
                print(msg.format(self.total_iteration + i, accu))
                self.step_list.append(self.total_iteration + i)
                self.accu_list.append(accu)
                self.cost_list.append(cost)
        self.total_iteration += num_iterations

        end_time = time.time()
        time_dif = end_time - start_time
        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        print("============One set of training: DONE!============\n\n")
        return None

    def getTrainingResults(self):
        '''
        Get some training results, which make it easy for plotting trends
        :return: step list, accuracy list, cost value list
        '''
        return self.step_list, self.accu_list, self.cost_list

    def getTestAccuracy(self, test_data, block_size = 128, n_cls_pred = 0):
        '''
        Compute accuracy using the testing examples
        Output certain number of predictions
        :param test_data: testing examples
        :param block_size: for Ram
        :param n_cls_pred: number of predictions that will be returned
        :return: accuracy value and a list of predictions
        '''
        n_test_example = len(test_data.labels)
        assert n_cls_pred <= block_size
        assert block_size <= n_test_example
        n_correct_pred = 0
        cls_pred = []
        i = 0
        while i < n_test_example:
            j = min(n_test_example, i + block_size)
            test_set = {
                self.__x: test_data.images[i: j],
                self.__y: test_data.labels[i: j],
            }
            tmp_pred = self.session.run(self.__y_cls_pred, feed_dict = test_set)
            n_correct_pred += (tmp_pred == test_data.cls[i: j]).sum()
            if i == 0:
                cls_pred = tmp_pred[: n_cls_pred]
            i = j
        return 1.*n_correct_pred/n_test_example, cls_pred


class CheckingParameters:
    '''
    For selecting the optimal parameters
    '''
    __naive_CNN_model = None

    def __init__(self, img_shape, num_cls, cs_list, cp_list, cr_list, fcs_list, fcr_list):
        '''
        Inialization of the test class
        :param img_shape: image shape
        :param num_cls: number of classes
        :param cs_list: conv layer shape list
        :param cp_list: conv layer pooling list
        :param cr_list: conv layer relu list
        :param fcs_list: fc layer shape list
        :param fcr_list: fc layer relu list
        '''
        print("============START ADJUSTING!===========\n\n")
        self.__naive_CNN_model = MultiConvLayerCNN(img_shape, num_cls)
        self.__naive_CNN_model.constuctCNNetwork(cs_list, cp_list, cr_list, fcs_list, fcr_list)

    def modifyModel(self, cs_list, cp_list, cr_list, fcs_list, fcr_list):
        '''
        In case that we need to change the Network structure
        :param cs_list: conv layer shape list
        :param cp_list: conv layer pooling list
        :param cr_list: conv layer relu list
        :param fcs_list: fc layer shape list
        :param fcr_list: fc layer relu list
        :return: None
        '''
        self.__naive_CNN_model.constuctCNNetwork(cs_list, cp_list, cr_list, fcs_list, fcr_list)
        return None

    def checkLearningRate(self, train_data, test_data, lower, upper, ratio, n_steps, batch_size = 64, opt_type = 'Adam'):
        '''
        For selecting the optimal learning rate
        :param lower: lower bound
        :param upper: upper bound
        :param ratio: increasing ratio
        :param n_steps: number of steps for test
        :param opt_type: optimizer type
        :return: list of learning rates, list of testing accuracy values
        '''
        print("============Adjusting Learning Rate:============\n")
        l_rate = lower
        l_rate_list = []
        accu_list = []
        while l_rate < upper:
            self.__naive_CNN_model.initialization(l_rate, opt_type = opt_type)
            self.__naive_CNN_model.trainData(train_data, num_iterations = n_steps, batch_size = batch_size)
            accuracy, _ = self.__naive_CNN_model.getTestAccuracy(test_data)
            print("Learning Rate: {0};\t\t Testing Accuracy: {1:>6.2%}".format(l_rate, accuracy))
            l_rate_list.append(l_rate)
            accu_list.append(accuracy)
            l_rate *= ratio

        # Print messages and reture
        print("============DONE!============\n\n")
        return l_rate_list, accu_list







