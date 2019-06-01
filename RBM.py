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

# CDK
def GibbsSampling(xx, hh, count, k):
    xk = sampleInt(tf.sigmoid(tf.matmul(W, hh) + tf.tile(c, [1, size_bt])))
    hk = sampleInt(tf.sigmoid(tf.matmul(tf.transpose(W), xk) + tf.tile(b, [1, size_bt])))
    return xk, hk, count+1, k

def checkIflessThanK(xk, hk, count, k):
    return count <= k

# size_h is the size of the hidden layer
size_h = 100
b, c, h, W, v, a = init_variable(size_h)
[xk1, hk1, _, _] = tf.while_loop(checkIflessThanK, GibbsSampling, [v, h, ct, k])
W_, b_, c_ = update_variable(a, v, h, xk1, hk1)
weight_offset = [W.assign_add(W_), b.assign_add(b_), c.assign_add(c_)]

# loop with batch
def rbm_image(weight_offset, data):
    
    # run session
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    
    tr_x = np.zeros((28,28))
    
    for i in range(1, 10002):
        tr_x, tr_y  = data.next_batch(size_bt)
        tr_x = np.transpose(tr_x)
        tr_y = np.transpose(tr_y)
        alpha = min(0.05, 100/float(i))
        sess.run(weight_offset, feed_dict={v: tr_x, a: alpha})
    
    image_gen = tile_raster_images(np.transpose(tr_x), img_shape=(28, 28), tile_shape=(10, 10),tile_spacing=(2, 2))
    imagex = Image.fromarray(image_gen)
    plt.imshow(imagex)
    plt.show()
    
print("Images generated using Training Dataset")
rbm_image(weight_offset, mnist.train)
print("Images generated using Testing Dataset")
rbm_image(weight_offset, mnist.test)
print("Images generated using Testing Dataset with 20% pixels removed")
rbm_image(weight_offset, testing_dataset_reduced_20)
print("Images generated using Testing Dataset with 50% pixels removed")
rbm_image(weight_offset, testing_dataset_reduced_50)
