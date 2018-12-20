"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""
import gym, os, sys, json
import numpy as np
import tensorflow as tf
from advance_DQN import DQN
from dqn_networks import build_dqn_example, build_dqn_dense
from utilRIL import initial_deepq_network, train_deepq_network


np.random.seed(1)
tf.set_random_seed(1)

def model_status( observation):
    return observation

def env_action( action):
    f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
    return np.array( np.array([f_action]))

def model_reward( reward):
    return reward/10. 


# python3 run_Pendulum.py train Pendulum.json Pendulum 10000 3 true

if __name__ == "__main__":
    global ACTION_SPACE
    argv=sys.argv
    cmd= argv[1]
    cfg_fn= argv[2]
    modelName= argv[3]
    
    if cmd=='train':
        max_steps= int(argv[4])
        episodes= int(argv[5])
        show= False
        if argv[6]=='true':
            show= True
        #jcfg, env, dqn = initial_deepq_network( cfg_fn, build_dqn_example)
        jcfg, env, dqn = initial_deepq_network( cfg_fn, build_dqn_dense)
        ACTION_SPACE= dqn.n_actions
        print(ACTION_SPACE)
        train_deepq_network(env, dqn, model_status, env_action, model_reward, max_steps, episodes, 'checkpoints/'+modelName, show )
        #train_dqn( cfg_fn, episodes, model, show)
        
        
    
    
    
