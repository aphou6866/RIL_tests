"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""
import gym, os, sys, json
import numpy as np
import tensorflow as tf
from ddpg_networks import build_ddpg_example, build_ddpg_dense, build_ddpg_BipedalWalker1
from utilRIL import initial_ddpg_method, ddpg_method_train

# python3 BipedalWalker.py train BipedalWalker.json BipedalWalker 5000 2000 100


def model_status( observation):
    #print( observation)
    return observation


def env_action( action, reward):
    global var
    if reward == None or reward < -100:
        var= 3
    else:
        var = max([var*0.9999, 0.01])  # add randomness to action selection for exploration
        
    a = np.clip(np.random.normal( action, var), -1, 1)    # add randomness to action selection for exploration
    return a
    #return np.clip(np.random.normal(action, 1), -1, 1)


def model_reward( reward):
    if reward== -100:
        reward= -2
    return reward

a=-200
b=300
def check_steps( max_steps, reward):
    if reward != None:
        #print( "update_learning_rate ", reward)
        #r= 1.0- ( reward - a)/ (b-a)
        #t= max_steps*( reward-a)/ (b-a)
        #print(t, reward)
        #if t<2000:
            #t= 500
        #print(r)
        if reward < 0:
            t=500
        elif reward < 10:
            t=1000
        else:
            t= max_steps
        return t
    return max_steps


# python3 BipedalWalker.py train BipedalWalker.json BipedalWalker 2000 20000 100

if __name__ == "__main__":
    argv=sys.argv
    cmd= argv[1]
    cfg_fn= argv[2]
    modelName= argv[3]
    
    if cmd=='train':
        max_steps= int(argv[4])
        episodes= int(argv[5])
        show= int(argv[6])
        jcfg, env, dqn = initial_ddpg_method( cfg_fn, build_ddpg_BipedalWalker1)
        ddpg_method_train(env, dqn, model_status, env_action, model_reward, check_steps, max_steps, episodes, 'checkpoints/'+modelName, show )
