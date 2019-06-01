# -*- coding: utf-8 -*-

import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import random
import copy
from tensorflow.examples.tutorials.mnist import input_data
from util import tile_raster_images

#### we do the first test on the minst data again
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

k = tf.constant(1)
ct = tf.constant(1)
    
# size_x is the size of the visible layer
size_x = 28*28
size_bt = 100 # batch size

def removePixels(dataset, ratio):
    max_count = int(784*(ratio/100))
    rindex = random.sample(range(0, 783), max_count)
    for i in range(len(dataset)):        
        for j in rindex:
            dataset[i][j] = 0
            
    return dataset

# create reduced testing dataset
testing_dataset_reduced_20 = copy.deepcopy(mnist.test)
removePixels(testing_dataset_reduced_20.images, 20)
testing_dataset_reduced_50 = copy.deepcopy(mnist.test)
removePixels(testing_dataset_reduced_50.images, 50)
testing_dataset_reduced_80 = copy.deepcopy(mnist.test)
removePixels(testing_dataset_reduced_80.images, 80)

# helper function
def sample(probs):
    return tf.to_float(tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1)))

def sampleInt(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

def init_variable(size_h):
    
    # RBM energy function
    # variables and place holder
    b = tf.Variable(tf.random_uniform([size_h, 1], -0.005, 0.005))
    c = tf.Variable(tf.random_uniform([size_x, 1], -0.005, 0.005))
    
    W = tf.Variable(tf.random_uniform([size_x, size_h], -0.005, 0.005))
    v = tf.placeholder(tf.float32, [size_x, size_bt])
    h = sample(tf.sigmoid(tf.matmul(tf.transpose(W), v) + tf.tile(b, [1, size_bt])))
    a = tf.placeholder(tf.float32)
    
    return b, c, h, W, v, a

def update_variable(a, x, h, xk1, hk1):
    
    w = tf.multiply(a/float(size_bt), tf.subtract(tf.matmul(x, tf.transpose(h)), tf.matmul(xk1, tf.transpose(hk1))))
    b = tf.multiply(a/float(size_bt), tf.reduce_sum(tf.subtract(h, hk1), 1, True))
    c = tf.multiply(a/float(size_bt), tf.reduce_sum(tf.subtract(x, xk1), 1, True))
    
    return w, b, c

