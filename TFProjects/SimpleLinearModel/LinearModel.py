import numpy as np
import tensorflow as tf

class LinearModel:
    '''
    We only add one hidden layer between
    the input layer and the output layer
    one inner product + one softmax
    '''

    __optimizer_list = {
        'GradDesc' : tf.train.GradientDescentOptimizer,
    }

    def __init__(self, x_dim, y_dim, l_rate = 0.5, opt_type = 'GradDesc'):
        '''
        Initialize the model
        :param x_dim: dimension of the input feature
        :param y_dim: dimension of the output, or the number of classes
        :param l_rate: learning rate
        :param opt_type: optimizer type
        '''

        # Define placeholder variables
        self.x = tf.placeholder(tf.float32, [None, x_dim])
        self.y = tf.placeholder(tf.float32, [None, y_dim])
        self.y_cls = tf.placeholder(tf.int64, [None])

        # Define variables to be optimized
        self.weights = tf.Variable(tf.random_normal((x_dim, y_dim)))
        self.biases = tf.Variable(tf.random_normal([y_dim]))

        # Define Model
        self.logits = tf.matmul(self.x, self.weights) + self.biases
        self.y_pred = tf.nn.softmax(self.logits)
        self.y_cls_pred = tf.argmax(self.y_pred, axis = 1)

        # Define Entropy or Lost Function or Cost function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
             logits = self.logits, labels = self.y)
        self.cost = tf.reduce_mean(cross_entropy)

        # Define Optimizer and Optimization variables
        self.learning_rate = l_rate
        assert opt_type in self.__optimizer_list.keys()
        self.optimizer = self.__optimizer_list[opt_type](
            learning_rate = self.learning_rate).minimize(self.cost)

        # Define Performance measurement
        correct_predictions = tf.equal(self.y_cls_pred, self.y_cls)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        # Create a Tensorflow session
        self.session = tf.Session()

    def initialization(self):
        '''
        Initialization of the model
        :return: None
        '''
        self.session.run(tf.global_variables_initializer())

    def modifyLearningRate(self, l_rate, opt_type = 'GradDesc'):
        '''
        In case if we need to modify the learning rate
        :param l_rate: new learning rate
        :param opt_type: new optimizer type
        :return: None
        '''
        self.learning_rate = l_rate
        self.optimizer = self.__optimizer_list[opt_type](
            learning_rate=self.learning_rate).minimize(self.cost)


    def trainingOptimize(self, data, num_iterations, batch_size):
        '''
        Train the model
        :param data: training data set
        :param num_iterations: number of iterations
        :param batch_size: batch size
        :return: a list of cost function values
        '''
        cost_list = []
        for i in range(num_iterations):
            x_batch, y_batch = data.next_batch(batch_size = batch_size)
            '''These batches are input for the iterations'''
            train_set = {
                self.x: x_batch,
                self.y: y_batch,
            }
            # Run the optimization
            cost_val, _ = self.session.run([self.cost, self.optimizer],
                                           feed_dict = train_set)
            cost_list.append(cost_val)
        return cost_list

    def getWeights(self):
        '''
        Get weights
        :return: weights
        '''
        return self.session.run(self.weights)

    def getTestAccuracy(self, test_data):
        '''
        Evaluate the accuracy using testing set
        :param test_data: testing data
        :return: Accuracy for the testing set
        '''
        test_set = {
            self.x: test_data.images,
            self.y: test_data.labels,
            self.y_cls: test_data.cls,
        }
        return self.session.run(self.accuracy, feed_dict = test_set)

    def getTestCost(self, test_data):
        '''
        Evaluate the Cost function using testing set
        :param test_data: testing data
        :return: Cost function computed from the testing set
        '''
        test_set = {
            self.x: test_data.images,
            self.y: test_data.labels,
        }
        return self.session.run(self.cost, feed_dict = test_set)

    def getTestPredictions(self, test_data):
        '''
        Get predictions for the test
        :param test_data: testing data
        :return: list of predictions corresponding to each test examples
        '''
        test_set = {
            self.x: test_data.images,
            self.y: test_data.labels,
        }
        return self.session.run(self.y_cls_pred, feed_dict=test_set)








