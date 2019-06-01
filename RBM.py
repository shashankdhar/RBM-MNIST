# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:13:26 2019

@author: shash
"""

import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import random
import copy
from tensorflow.examples.tutorials.mnist import input_data

#### we do the first test on the minst data again
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

k = tf.constant(1)
ct = tf.constant(1)
    
# size_x is the size of the visiable layer
size_x = 28*28
size_bt = 100 # batch size

# helper function
def sample(probs):
    return tf.to_float(tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1)))

def sampleInt(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

