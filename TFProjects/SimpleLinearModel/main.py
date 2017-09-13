# Add '../' into python path, so that we don't need download the data over and over again
import sys
sys.path.append('../')


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data