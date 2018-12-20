"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
from networks import fc, dense

'''
import tensorflow as tf
from dqn_networks import build_dqn_dense
X= tf.placeholder(tf.float32, [None, 5], name='input')
out= build_dqn_dense(X, None, 3, 5, None, None, True, True)
g= tf.get_default_graph()
tf.summary.FileWriter('logs',g).close()

!tensorboard --logdir=logs

'''
 
#def build_dqn_example( X, n_action, dueling, trainable=False):
    #l1= dense( X, activation= tf.nn.relu, trainable=trainable)
        
    #if dueling:
        ## Dueling DQN
        #V= fc( l1, 'Value', nh=1, init_scale=np.sqrt(2), trainable=trainable)
        #A= fc( l1, 'Advantage', nh= n_action, init_scale=np.sqrt(2), trainable=trainable)
        #with tf.variable_scope('Q'):
            #out = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True), trainable=trainable)     # Q = V(s) + A(s,a)
    #else:
        #out= fc( l1, 'Q', nh=n_action, init_scale=np.sqrt(2), trainable=trainable)
            
    #return out

def build_dqn_dense( X, c_names, n_actions, layers, hiddens, w_initializer, b_initializer, dueling, trainable):
    print('----------->',layers, hiddens, dueling)
    l1= dense( X, num_layers=layers, num_hidden=hiddens, activation= tf.nn.relu, c_names= c_names)
    if dueling:
        V= fc( l1, 'Value', nh=1, init_scale=np.sqrt(2), c_names= c_names)
        A= fc( l1, 'Advantage', nh= n_actions, init_scale=np.sqrt(2), c_names= c_names)
        with tf.variable_scope('Q'):
            out = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
    else:
        out= fc( l1, 'Q', nh=n_actions, init_scale=np.sqrt(2), c_names= c_names)
    return out


def build_dqn_example(s, c_names, n_actions, layers, hiddens, w_initializer, b_initializer, dueling, trainable):
    n_features= s.get_shape()[1].value
    with tf.variable_scope('l1'):
        w1 = tf.get_variable('w1', [n_features, hiddens], initializer=w_initializer, collections=c_names, trainable=trainable)
        b1 = tf.get_variable('b1', [1, hiddens], initializer=b_initializer, collections=c_names,  trainable=trainable)
        l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
        
    if dueling:
        # Dueling DQN
        with tf.variable_scope('Value'):
            w2 = tf.get_variable('w2', [hiddens, 1], initializer=w_initializer, collections=c_names)
            b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
            V = tf.matmul(l1, w2) + b2

        with tf.variable_scope('Advantage'):
            w2 = tf.get_variable('w2', [hiddens, n_actions], initializer=w_initializer, collections=c_names)
            b2 = tf.get_variable('b2', [1, n_actions], initializer=b_initializer, collections=c_names)
            A = tf.matmul(l1, w2) + b2

        with tf.variable_scope('Q'):
            out = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
    else:
        with tf.variable_scope('Q'):
            w2 = tf.get_variable('w2', [hiddens, n_actions], initializer=w_initializer, collections=c_names)
            b2 = tf.get_variable('b2', [1, n_actions], initializer=b_initializer, collections=c_names)
            out = tf.matmul(l1, w2) + b2
            
    return out

