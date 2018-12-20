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
    global current_state
    current_state= observation
    #observation[1]= observation[1]/0.004
    #print(observation)
    return observation

def env_action( action):
    return action

def model_reward( reward):
    #print(reward, current_state)
    #print(observation_)
    reward= current_state[0]
    return reward


# python3 run_MountainCar.py train MountainCar.json  MountainCar 2000000 10 true

if __name__ == "__main__":
    
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
        train_deepq_network(env, dqn, model_status, env_action, model_reward, max_steps, episodes, 'checkpoints/'+modelName, show )
        #train_dqn( cfg_fn, episodes, model, show)
        
        
    
    
    
