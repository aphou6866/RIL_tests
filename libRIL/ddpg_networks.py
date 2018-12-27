"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym, time, os, sys
from networks import fc, dense

'''
import tensorflow as tf
from ddpg_networks import build_ddpg_example
s= tf.placeholder(tf.float32, [None, 5], name='input')
out= build_ddpg_example( s, 3, 0.1, 1, 30)
g= tf.get_default_graph()
tf.summary.FileWriter('logs',g).close()
exit()

tensorboard --logdir=logs

'''
'''
def build_ddpg_example( s, a_dim, a_bound, layers, hiddens, reuse=None, custom_getter=None):
    s_dim= s.get_shape()[1].value
    
    trainable = True if reuse is None else False
    with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
        net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
        a = tf.layers.dense(net, a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
        a_= tf.multiply(a, a_bound, name='scaled_a')
    
    with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
        n_l1 = 30
        w1_s = tf.get_variable('w1_s', [s_dim, n_l1], trainable=trainable)
        w1_a = tf.get_variable('w1_a', [a_dim, n_l1], trainable=trainable)
        b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
        net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul( a_, w1_a) + b1)
        q= tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
        
    return a_, q


def build_ddpg_dense( s, a_dim, a_bound, layers, hiddens, reuse=None, custom_getter=None):
    
    s_dim= s.get_shape()[1].value
    trainable = True if reuse is None else False
    print('layers, hiddens <==>',layers, hiddens)
    with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
        net= dense( s, num_layers=layers, num_hidden=hiddens, activation= tf.nn.relu)
        a = tf.layers.dense(net, a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
        a_= tf.multiply(a, a_bound, name='scaled_a')
        
    with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
        x= dense( s, num_layers=layers, num_hidden=hiddens, activation= tf.nn.relu)
        w1_s = tf.get_variable('w1_s', [hiddens, hiddens], trainable=trainable)
        w1_a = tf.get_variable('w1_a', [a_dim, hiddens], trainable=trainable)
        b1 = tf.get_variable('b1', [1, hiddens], trainable=trainable)
        net = tf.nn.relu(tf.matmul( x, w1_s) + tf.matmul( a_, w1_a) + b1)
        q= tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
        
    return a_, q

'''
def build_ddpg_example( s, a_dim, a_bound, layers, hiddens, scope, trainable, discrete):
    s_dim= s.get_shape()[1].value
    with tf.variable_scope(scope):
        #with tf.variable_scope('Prenet'):
            #p= tf.layers.dense(s, 30, activation=tf.nn.relu, name='l0', trainable=trainable)
        
        with tf.variable_scope('Actor'):
            net = tf.layers.dense( s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            a = tf.multiply(a, a_bound, name='scaled_a')
        
        with tf.variable_scope('Critic'):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul( s, w1_s) + tf.matmul(a, w1_a) + b1)
            q= tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    return a, q


def build_ddpg_dense( s, a_dim, a_bound, layers, hiddens, scope, trainable, discrete):
    print('layers, hiddens, discrete<==>',layers, hiddens, discrete)
    with tf.variable_scope(scope):
        with tf.variable_scope('Prenet'):
            x= dense( s, num_layers=layers, num_hidden=hiddens, activation= tf.nn.relu)
        
        with tf.variable_scope('Actor'):
            net = tf.layers.dense( x, hiddens, activation=tf.nn.relu, name='l1', trainable=trainable)
            if discrete== False:
                a = tf.layers.dense(net, a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
                a = tf.multiply(a, a_bound, name='scaled_a')
            else:
                a = tf.layers.dense(net, a_dim, activation=tf.nn.softmax, name='a', trainable=trainable)
        
        with tf.variable_scope('Critic'):
            w1_s = tf.get_variable('w1_s', [hiddens, hiddens], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [a_dim, hiddens], trainable=trainable)
            b1 = tf.get_variable('b1', [1, hiddens], trainable=trainable)
            net = tf.nn.relu(tf.matmul( x, w1_s) + tf.matmul(a, w1_a) + b1)
            q= tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    return a, q


def build_ddpg_BipedalWalker1( s, a_dim, a_bound, layers, hiddens, scope, trainable, discrete):
    s_dim= s.get_shape()[1].value
    with tf.variable_scope(scope):
        
        with tf.variable_scope('Actor'):
            init_w = tf.random_normal_initializer(0., 0.01)
            init_b = tf.constant_initializer(0.01)
            net = tf.layers.dense(s, 500, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2', trainable=trainable)

            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, a_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        
        with tf.variable_scope('Critic'):
            init_w = tf.random_normal_initializer(0., 0.01)
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 700
                # combine the action and states together in this way
                w1_s = tf.get_variable('w1_s', [s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul( scaled_a, w1_a) + b1)
            with tf.variable_scope('l2'):
                net = tf.layers.dense(net, 20, activation=tf.nn.relu, kernel_initializer=init_w,
                                      bias_initializer=init_b, name='l2', trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)

    return scaled_a, q
