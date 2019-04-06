import numpy as np
import tensorflow as tf


### LAYERS ###

def weight(shape):
    """ creates weights tensor of desired shape """
    scale = np.sqrt(2 / (np.prod(shape) - 1))
    return tf.Variable(tf.truncated_normal(shape, stddev=scale))

def bias(shape, value=0.0):
    """ creates bias tensor of desired shape """
    return tf.Variable(tf.constant(value, shape=shape))

def dense(x, shape, activation, with_weights=False):
    """ """
    if type(shape) is int:
        shape = (x.shape.as_list()[-1], shape)
    W = weight(shape)
    b = bias([shape[-1]])
    x_next = tf.matmul(x, W) + b
    if activation is not None:
        x_next = activation(x_next)
    if with_weights:
        return x_next, W, b
    return x_next

def dense_w(x, W, b, activation):
    """ """
    return activation(tf.matmul(x, W) + b)

### HELPER ###

def msg(string, print_me):
    """ """
    if print_me:
        print(string)


