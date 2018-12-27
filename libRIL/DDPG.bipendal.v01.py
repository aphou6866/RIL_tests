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
import gym, time, os, sys, json


###############################  Actor  ####################################

class Actor(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, t_replace_iter, GLOBAL_STEP):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        with tf.name_scope('S'):
            self.S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
        with tf.name_scope('S_'):
            self.S_= tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')
        self.GLOBAL_STEP= GLOBAL_STEP

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net( self.S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net( self.S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.01)
            init_b = tf.constant_initializer(0.01)
            net = tf.layers.dense(s, 500, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2', trainable=trainable)

            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):  # batch update
        self.sess.run(self.train_op, feed_dict={ self.S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={ self.S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # self.a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads_and_vars = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads_and_vars, self.e_params), global_step= self.GLOBAL_STEP)


###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, GLOBAL_STEP, s, s_, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        self.GLOBAL_STEP= GLOBAL_STEP
        with tf.name_scope('R'):
            self.R = tf.placeholder(tf.float32, [None, 1], name='r')
        self.S = s
        self.S_= s_

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(self.S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(self.S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = self.R + self.gamma * self.q_

        with tf.variable_scope('abs_TD'):
            self.abs_td = tf.abs(self.target_q - self.q)
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.GLOBAL_STEP)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.01)
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 700
                # combine the action and states together in this way
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            with tf.variable_scope('l2'):
                net = tf.layers.dense(net, 20, activation=tf.nn.relu, kernel_initializer=init_w,
                                      bias_initializer=init_b, name='l2', trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_, ISW):
        _, abs_td = self.sess.run([self.train_op, self.abs_td], feed_dict={self.S: s, self.a: a, self.R: r, self.S_: s_, self.ISWeights: ISW})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1
        return abs_td


class SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)+1e-5
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add_new_priority(self, p, data):
        leaf_idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data  # update data_frame
        self.update(leaf_idx, p)  # update tree_frame
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]

        self.tree[tree_idx] = p
        self._propagate_change(tree_idx, change)

    def _propagate_change(self, tree_idx, change):
        """change the sum of priority value in all parent nodes"""
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    def get_leaf(self, lower_bound):
        leaf_idx = self._retrieve(lower_bound)  # search the max leaf priority based on the lower_bound
        data_idx = leaf_idx - self.capacity + 1
        return [leaf_idx, self.tree[leaf_idx], self.data[data_idx]]

    def _retrieve(self, lower_bound, parent_idx=0):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        left_child_idx = 2 * parent_idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):  # end search when no more child
            return parent_idx

        if self.tree[left_child_idx] == self.tree[right_child_idx]:
            return self._retrieve(lower_bound, np.random.choice([left_child_idx, right_child_idx]))
        if lower_bound <= self.tree[left_child_idx]:  # downward search, always search for a higher priority node
            return self._retrieve(lower_bound, left_child_idx)
        else:
            return self._retrieve(lower_bound - self.tree[left_child_idx], right_child_idx)

    @property
    def root_priority(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.001  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 1e-5  # annealing the bias
    abs_err_upper = 1   # for stability refer to paper

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, error, transition):
        p = self._get_priority(error)
        self.tree.add_new_priority(p, transition)

    def prio_sample(self, n):
        batch_idx, batch_memory, ISWeights = [], [], []
        segment = self.tree.root_priority / n
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root_priority
        maxiwi = np.power(self.tree.capacity * min_prob, -self.beta)  # for later normalizing ISWeights
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            lower_bound = np.random.uniform(a, b)
            while True:
                idx, p, data = self.tree.get_leaf(lower_bound)
                if type(data) is int:
                    i -= 1
                    lower_bound = np.random.uniform(segment * i, segment * (i+1))
                else:
                    break
            prob = p / self.tree.root_priority
            ISWeights.append(self.tree.capacity * prob)
            batch_idx.append(idx)
            batch_memory.append(data)

        ISWeights = np.vstack(ISWeights)
        ISWeights = np.power(ISWeights, -self.beta) / maxiwi  # normalize
        return batch_idx, np.vstack(batch_memory), ISWeights

    def random_sample(self, n):
        idx = np.random.randint(0, self.tree.capacity, size=n, dtype=np.int)
        return np.vstack(self.tree.data[idx])

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _get_priority(self, error):
        error += self.epsilon   # avoid 0
        clipped_error = np.clip(error, 0, self.abs_err_upper)
        return np.power(clipped_error, self.alpha)



class ddpg_method(object):
    def __init__(self, build_network, layers, hiddens,
                    action_dim, state_dim, action_bound,
                    discrete, REPLACEMENT, LR_decay,
                    LR_A = 0.001,    # learning rate for actor
                    LR_C = 0.002,    # learning rate for critic
                    GAMMA = 0.9,     # reward discount
                    TAU = 0.01,      # soft replacement
                    MEMORY_CAPACITY = 10000,
                    BATCH_SIZE = 32,
                    OUTPUT_GRAPH= False
        ):
        np.random.seed(1)
        tf.set_random_seed(1)
        self.state_dim= state_dim
        self.action_dim= action_dim
        self.MEMORY_CAPACITY= MEMORY_CAPACITY
        self.BATCH_SIZE= BATCH_SIZE
        
        self.GLOBAL_STEP = tf.Variable(0, trainable=False)
        self.INCREASE_GS = self.GLOBAL_STEP.assign( tf.add( self.GLOBAL_STEP, 1))
        self.LR_A = tf.train.exponential_decay(LR_A, self.GLOBAL_STEP, LR_decay['decay_steps'], LR_decay['decay_rate'] , staircase=True)
        self.LR_C = tf.train.exponential_decay(LR_C, self.GLOBAL_STEP, LR_decay['decay_steps'], LR_decay['decay_rate'], staircase=True)
        self.sess = tf.Session()
        
        # Create actor and critic.
        self.actor = Actor( self.sess, state_dim, action_dim, action_bound, self.LR_A, REPLACEMENT['rep_iter_a'], self.GLOBAL_STEP)
        self.critic = Critic( self.sess, state_dim, action_dim, self.LR_C, GAMMA, REPLACEMENT['rep_iter_c'], self.GLOBAL_STEP, self.actor.S, self.actor.S_, self.actor.a, self.actor.a_)
        self.actor.add_grad_to_graph( self.critic.a_grads)

        self.saver= tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        self.M = M = Memory(MEMORY_CAPACITY)
        
        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)
    
    
    def learn(self):
        if self.GLOBAL_STEP.eval(self.sess) > self.MEMORY_CAPACITY/20:
            tree_idx, b_M, ISWeights = self.M.prio_sample( self.BATCH_SIZE)    # for critic update
            b_s = b_M[:, :self.state_dim]
            b_a = b_M[:, self.state_dim: self.state_dim + self.action_dim]
            b_r = b_M[:, -self.state_dim - 1: -self.state_dim]
            b_s_ = b_M[:, -self.state_dim:]

            abs_td = self.critic.learn(b_s, b_a, b_r, b_s_, ISWeights)
            self.actor.learn(b_s)
            for i in range(len(tree_idx)):  # update priority
                idx = tree_idx[i]
                self.M.update(idx, abs_td[i])

        
    def load(self, modelName):
        jfn= "./"+modelName+".json"
        fn = "./"+modelName+".ckpt"
        if os.path.isfile(jfn):
            with open(jfn) as f:
                info= json.load(f)
            self.saver.restore(self.sess, fn)
            print("Load model:", fn, 'with reward', info['reward'])
            return info['reward']
        return float('-inf')

    def save(self, modelName, reward):
        info= {'reward': reward}
        jfn= "./"+modelName+".json"
        save_path = self.saver.save(self.sess, "./"+modelName+".ckpt")
        with open( jfn, "w") as f:
            f.write('%s'%json.dumps(info) )
            f.write('\n')
            
        print('Save model ', modelName, " to path", save_path, 'with reward', info['reward'])
        
