import numpy as np
import tensorflow as tf
import time
from datetime import timedelta

class TwoConvCNN:
    '''
    In this model, we only consider adding two convlutional layers
    +two fully connected layers
    Current code is only for img_classifications
    '''

    __optimizer_list = {
        'GradDesc': tf.train.GradientDescentOptimizer,
        'Adam': tf.train.AdamOptimizer,
        'AdGrad': tf.train.AdagradOptimizer,
    }

    def __init__(self, img_shape, num_cls):
        '''
        Initialize the class
        :param img_shape: image dimensions = [height, width, number of channels]
        :param num_cls: number of classes
        '''
        # Define placeholder variables
        self.img_shape = img_shape
        self.x = tf.placeholder(tf.float32, [None, np.array(img_shape).prod()])
        self.x_img = tf.reshape(self.x, [-1] + list(img_shape))
        self.y = tf.placeholder(tf.float32, [None, num_cls])
        self.y_cls = tf.argmax(self.y, axis = 1)
        self.conv1 = None
        self.weight1 = None
        self.conv2 = None
        self.weight2 = None
        self.fc1 = None
        self.wfc1 = None
        self.wfc2 = None
        self.fc2 = None

    def createConvLayer(self, input, layer_shape, num_filters, pooling = True):
        '''
        Create a Convolutional layer
        :param input: output of the previous layer
        :param layer_shape: layer dimensions = [filter_height, filter_width, filter_depth]
        :param num_filters: number of filters
        :param pooling: whether use puoling or down sampling
        :return: the convolutional layer
        '''
        # Construct weights and biases
        shape = list(layer_shape) + [num_filters]
        weights = tf.Variable(tf.truncated_normal(shape, mean = 0., stddev = 0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

        ## Use zero padding (padding = "SAME")
        ## Stride : 1 X 1
        layer = tf.nn.conv2d(input = input, filter = weights,
                             strides = [1, 1, 1, 1], padding = 'SAME')
        layer += biases

        # Determine whether do pulling (down sampling)
        # Here we only use a 2X2 down sample pulling
        if pooling :
            layer = tf.nn.max_pool(value = layer, ksize = [1, 2, 2, 1],
                                   strides = [1, 2, 2, 1], padding = 'SAME')
        layer = tf.nn.relu(layer)

        # return layer + weights
        print('Create a convolutional layer: DONE!')
        return layer, weights

    def flattenLayer(self, input_layer):
        '''
        Flatten the last convolutioanal layer
        :param input_layer: the last convolutional layer
        :return: the flatten layer
        '''
        shape = input_layer.get_shape().as_list()
        num_features = np.array(shape[1:4], dtype = np.int64).prod()

        # Create A new Flat layer
        layer_flat = tf.reshape(input_layer, [-1, num_features])

        # return layer and the corresponding dimension
        print("Create a flatten layer: DONE!")
        return layer_flat, num_features

    def createFcLayer(self, input, fc_shape, relu = True):
        '''
        Create a fully connected layer
        :param input: the previous layer (must be flattened)
        :param fc_shape: the dimensions [number of input, number of output]
        :param relu: whether to use relu
        :return: The fully connected layers
        '''
        weights = tf.Variable(tf.truncated_normal(fc_shape, mean = 0., stddev = 0.05))
        biases = tf.Variable(tf.constant(0.05, shape = fc_shape[1:]))

        output = tf.matmul(input, weights) + biases

        if relu:
            output = tf.nn.relu(output)

        # return the output layer
        print("Create a fully connected layer: DONE!")
        return output, weights

    def constructModel(self, conv1_shape, conv2_shape, fc1_shape, fc2_shape):
        '''
        Construct the model
        :param conv1_shape: [filter_height, filter_size, filter_depth, number of filters]
        :param conv2_shape: [filter_height, filter_size, filter_depth, number of filters]
        :param fc1_shape: [fc_input_size, fc_output_size]
        :param fc2_shape: [fc_input_size, fc_output_size]
        :return: None
        '''
        self.conv1, self.weight1 = self.createConvLayer(
            self.x_img, conv1_shape[0:3], conv1_shape[3], pooling = True)

        assert conv1_shape[3] == conv2_shape[2]
        self.conv2, self.weight2 = self.createConvLayer(
            self.conv1, conv2_shape[0:3], conv2_shape[3], pooling = True)

        flat_layer, num_features = self.flattenLayer(self.conv2)

        fc1_shape[0] = num_features
        self.fc1, self.wfc1 = self.createFcLayer(flat_layer, fc1_shape, relu = True)

        assert fc1_shape[1] == fc2_shape[0]
        self.fc2, self.wfc2 = self.createFcLayer(self.fc1, fc2_shape, relu = True)

        # Target variables
        self.y_pred = tf.nn.softmax(self.fc2)
        self.y_cls_pred = tf.argmax(self.y_pred, axis = 1)

        # return None
        print("Construct the model: DONE!")
        return None

    def initialization(self, l_rate, opt_type = 'GradDesc'):
        '''
        Initialization and start a tf session
        :param l_rate: learning rate
        :param opt_type: type of the optimizer
        :return: None
        '''
        # Define Entropy or Lost Function or Cost function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.fc2, labels=self.y)
        self.cost = tf.reduce_mean(cross_entropy)

        # Define Optimizer and Optimization variables
        assert opt_type in self.__optimizer_list.keys()
        self.optimizer = self.__optimizer_list[opt_type](
            learning_rate=l_rate).minimize(self.cost)

        # Define Performance measurement
        correct_predictions = tf.equal(self.y_cls_pred, self.y_cls)
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
        print("Ready for training: DONE!")
        return None

    def training(self, data, num_iterations, batch_size):
        '''
        Training the model
        :param data: training set
        :param num_iterations: number of iterations
        :param batch_size: batch_size
        :return: None
        '''
        start_time = time.time()
        for i in range(num_iterations):

            # Get training set
            x_batch, y_batch = data.next_batch(batch_size)
            train_set = {
                self.x: x_batch,
                self.y: y_batch,
            }
            self.session.run(self.optimizer, feed_dict = train_set)
            if self.total_iteration%50 == 0:
                accu, cost = self.session.run([self.accuracy, self.cost], feed_dict = train_set)

                # Message for printing
                msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
                print(msg.format(self.total_iteration, accu))
                self.step_list.append(self.total_iteration)
                self.accu_list.append(accu)
                self.cost_list.append(cost)
            self.total_iteration += 1
        print("One set of training: DONE!")

        end_time = time.time()
        time_dif = end_time - start_time
        # Print the time-usage.
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        return None

    def getTrainingResults(self):
        '''
        Get training results
        :return: steps, accuracy, cost values
        '''
        return self.step_list, self.accu_list, self.cost_list

    def getTestAccuracy(self, test_data):
        '''
        Compute the accuracy
        :param test_data: test data set
        :return: accuracy for the test set, prediction for the test set
        '''
        batch_size = 256
        num_test_example = len(test_data.labels)
        cls_pred = np.zeros(shape = num_test_example, dtype = np.int)
        i = 0
        while i < num_test_example:
            j = min(i+batch_size, num_test_example)
            test_set = {
                self.x: test_data.images[i: j],
                self.y: test_data.labels[i: j],
            }
            cls_pred[i: j] = self.session.run(self.y_cls_pred, feed_dict = test_set)
            i = j

        correct = (cls_pred == test_data.cls)
        return 1.*correct.sum()/num_test_example, cls_pred








