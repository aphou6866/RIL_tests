"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""
import gym, os, sys, json
import numpy as np
import tensorflow as tf
from ddpg_networks import build_ddpg_example, build_ddpg_dense
from utilRIL import initial_ddpg_method, ddpg_method_train

# python3 Pendulum.py train Pendulum.json Pendulum 200 200 -300

def model_status( observation):
    return observation

def env_action( action, reward):
    global var
    if reward == None or reward < -100:
        var= 3
    else:
        var = var*0.9999
    
    a = np.clip(np.random.normal( action, var), -2, 2) 
    return action

def model_reward( reward):
    return reward

def check_steps( max_steps, reward):
    return max_steps
# python3 Pendulum.py train Pendulum.json Pendulum 200 200 true


if __name__ == "__main__":
    argv=sys.argv
    cmd= argv[1]
    cfg_fn= argv[2]
    modelName= argv[3]
    
    if cmd=='train':
        max_steps= int(argv[4])
        episodes= int(argv[5])
        show= int(argv[6])
        jcfg, env, dqn = initial_ddpg_method( cfg_fn, build_ddpg_example)
        ddpg_method_train(env, dqn, model_status, env_action, model_reward, check_steps, max_steps, episodes, 'checkpoints/'+modelName, show )
