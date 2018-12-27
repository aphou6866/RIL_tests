import pandas as pds
import numpy as np
import tensorflow as tf

def build_dense_net( n, m, layers, hiddens, scope):
    
    X= tf.placeholder(tf.float32, [None, n], 'X')
    with tf.variable_scope(scope):
        for i in range( layers):
            w_init = tf.random_normal_initializer(0., .1)
            if i==0:
                net = tf.layers.dense( X, hiddens, tf.nn.relu, kernel_initializer=w_init, name='hidden%02d'%i)
            else:
                net = tf.layers.dense( net, hiddens, tf.nn.relu, kernel_initializer=w_init, name='hidden%02d'%i)
        y= tf.layers.dense( net, m, tf.nn.relu, kernel_initializer=w_init, name='y')
        y_train= pred_softmax = tf.nn.softmax(y, name='output')
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    return X, y_train, params

