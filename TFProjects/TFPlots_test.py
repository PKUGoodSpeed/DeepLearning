import TFPlots
import unittest
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class MnistImagePlot_test(unittest.TestCase):
    """Set of tests to check class behavior"""

    def setUp(self):
        self.mnist_plot = TFPlots.MnistImagePlot(height = 28, width = 28)

    def plotImage_test(self):
        data = input_data.read_data_sets('data/MNIST/', one_hot = True)

        r = 3
        c = 4

        imgs = data.test.images[0: r * c]
        cls = np.argmax(data.test.labels, axis = 1)[0 : r * c]

        self.mnist_plot.plotImage(r, c, imgs, cls)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(MnistImagePlot_test)
    unittest.TextTestRunner(verbosity = 2).run(suite)


    ## Show Figures
    data = input_data.read_data_sets('data/MNIST/', one_hot=True)
    r = 3
    c = 4

    imgs = data.test.images[0: r * c]
    cls = np.argmax(data.test.labels, axis=1)[0: r * c]

    TFPlots.MnistImagePlot().plotImage(r, c, imgs, cls)