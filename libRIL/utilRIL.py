"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""
import gym, json
import numpy as np
import tensorflow as tf
from advance_DQN import DQN

def initial_deepq_network( fn, build_network):
    with open(fn) as f:
        jcfg= json.load(f)
    print( json.dumps(jcfg, sort_keys=True, indent=4, separators=(',',':')))
    #print(type(jcfg['DQN']['double_q']))
    env = gym.make(jcfg['environment']['name'])
    env = env.unwrapped
    return jcfg, env, DQN(
                build_network,
                jcfg['DQN']['n_actions'],
                jcfg['DQN']['n_features'],
                learning_rate= jcfg['DQN']['learning_rate'],
                reward_decay= jcfg['DQN']['reward_decay'],
                e_greedy= jcfg['DQN']['e_greedy'],
                replace_target_iter= jcfg['DQN']['replace_target_iter'],
                memory_size= jcfg['DQN']['memory_size'],
                batch_size= jcfg['DQN']['batch_size'],
                e_greedy_increment= jcfg['DQN']['e_greedy_increment'],
                output_graph= jcfg['DQN']['output_graph'],
                layers= jcfg['DQN']['layers'],
                hiddens= jcfg['DQN']['hiddens'],
                prioritized= jcfg['DQN']['prioritized'],
                double_q= jcfg['DQN']['double_q'],
                dueling= jcfg['DQN']['dueling'],
                sess=None,
        )


def train_deepq_network( env, model,  model_status, env_action, model_reward, max_steps, episodes, modelName, show):
    
    model.load( modelName)
    total_steps = 0
    for i_episode in range(episodes):
        observation = model_status( env.reset())
        ep_r = 0
        steps= 0
        while True:
            if show:
                env.render()
            action = model.choose_action(observation)
            observation_, reward, done, info = env.step( env_action(action))
            observation_ = model_status( observation_)
            reward= model_reward( reward)
            model.store_transition(observation, action, reward, observation_)

            ep_r += reward
            if 4< total_steps:
                model.learn()
            if total_steps%100==3:
                print(total_steps," avgRwd:", ep_r/ steps)
            if done or max_steps< total_steps:
                print('episode: ', i_episode,
                    'ep_r: ', round(ep_r, 2),
                    ' epsilon: ', round(model.epsilon, 2))
                break

            observation = observation_
            total_steps += 1
            steps +=1 
            
        if max_steps < total_steps:
            break;

    model.save(modelName)
    
#def train_deepq_network( env, brain, reward_scale, episodes, model, show):
    
    #continuous=False
    #print(type(env.action_space), env.action_space, len(env.action_space.shape))
    
    #if len(env.action_space.shape)==1:
        #continuous= True
        #A_BOUND = [env.action_space.low, env.action_space.high]
        #A_LENGTH= env.action_space.high- env.action_space.low
        #print( A_BOUND[0], A_BOUND[1], A_LENGTH)
    
    #for i_episode in range(episodes):
        #observation = env.reset()
        #total_steps =0
        #sumRwd=0.
        #mntRwd=0.
        #while True:
            #if show:
                #env.render()
                
            #action = brain.choose_action( observation)
            #if continuous:
                #t= action/ brain.n_actions
                #f_action= A_BOUND[0] + t* A_LENGTH
            #else:
                #f_action= action
            ##f_action = (action-( brain.n_actions-1)/2)/(( brain.n_actions-1)/4)
            ##print(action, observation)
            #observation_, reward, done, info = env.step(f_action)
            #sumRwd += reward
            #if 3< total_steps:
                #avgRwd= sumRwd/ total_steps
            #else:
                #avgRwd= 0.0
            #reward *= reward_scale
            ##print( action, observation, reward, R_BOUND)
            #if total_steps%1000==0:
                #print( total_steps, avgRwd)
            #brain.store_transition(observation, action, reward, observation_)
            
            #if total_steps > brain.memory_size:   # learning
                #brain.learn()
            #if done:
                #print("Game done")
                #break
            
            #observation = observation_
            #total_steps += 1
    
    
