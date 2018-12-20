import os
import numpy as np
import tensorflow as tf
from collections import deque


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [nf] if one_dim_bias else [1, nf, 1, 1]
    nin = x.get_shape()[channel_ax].value
    wshape = [rf, rf, nin, nf]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            b = tf.reshape(b, bshape)
        return tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format) + b

def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0, c_names=None):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        if c_names !=None:
            w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale), collections=c_names)
            b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias), collections=c_names)
        else:
            w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
            b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b


def dense( X, num_layers=1, num_hidden=20, activation=tf.tanh, layer_norm=False, c_names=None):
    h = tf.layers.flatten(X)
    for i in range(num_layers):
        h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2), c_names=c_names)
        if layer_norm:
            h = tf.contrib.layers.layer_norm(h, center=True, scale=True, collections=c_names)
        h = activation(h)

    return h
