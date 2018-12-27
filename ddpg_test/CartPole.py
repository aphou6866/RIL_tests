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
    global current_state
    current_state= observation
    return observation

def env_action( action, sampling):
    #if sampling:
        #a = np.random.choice(action.shape[0], p= action)
    #else:
        #a = np.argmax(action)
    #print(a, action)
    #return a
    return np.random.choice(action.shape[0], p= action)

def model_reward( reward):
    observation_= current_state
    x, x_dot, theta, theta_dot = observation_
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward



# python3 CartPole.py train CartPole.json CartPole 300 2000 200

if __name__ == "__main__":
    argv=sys.argv
    cmd= argv[1]
    cfg_fn= argv[2]
    modelName= argv[3]
    
    if cmd=='train':
        max_steps= int(argv[4])
        episodes= int(argv[5])
        show= int(argv[6])
        jcfg, env, dqn = initial_continuous_ddpg( cfg_fn, build_ddpg_dense)
        train_continuous_ddpg(env, dqn, model_status, env_action, model_reward, max_steps, episodes, 'checkpoints/'+modelName, show  )
