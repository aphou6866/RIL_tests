"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""
import gym, os, sys, json
import numpy as np
import tensorflow as tf
from DDPG import continuous_DDPG, continuous_DDPG_update2
from ddpg_networks import build_ddpg_example, build_ddpg_dense
from utilRIL import initial_continuous_ddpg, train_continuous_ddpg

# python3 Pendulum.py train Pendulum.json Pendulum 200 200 true

def model_status( observation):
    return observation

def env_action( action, sampling):
    if sampling==True:
        a= np.clip(np.random.normal(action, 3), -1, 1)
    else:
        a= action
    return a

def model_reward( reward):
    return reward


# python3 MountainCar.py train MountainCar.json MountainCar 200 200 true


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
        jcfg, env, dqn = initial_continuous_ddpg( cfg_fn, build_ddpg_dense)
        train_continuous_ddpg(env, dqn, model_status, env_action, model_reward, max_steps, episodes, 'checkpoints/'+modelName, -1000 )
