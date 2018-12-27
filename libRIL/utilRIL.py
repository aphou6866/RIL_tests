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
from DDPG import ddpg_method
from numpy import linalg as LA

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
    
    
def initial_ddpg_method( fn, build_network):
    with open(fn) as f:
        jcfg= json.load(f)
    print( json.dumps(jcfg, sort_keys=True, indent=4, separators=(',',':')))
    #print(type(jcfg['DQN']['double_q']))
    env = gym.make(jcfg['environment']['name'])
    env = env.unwrapped
    env.seed(1)
    s_dim = env.observation_space.shape[0]
    if jcfg['DDPG']['discrete']:
        a_dim = env.action_space.n
        a_bound=0
    else:
        a_dim = env.action_space.shape[0]
        a_bound = env.action_space.high
    print("s_dim, a_dim, a_bound", s_dim, a_dim, a_bound)
    return jcfg, env, ddpg_method(
                    build_network, 
                    layers= jcfg['DDPG']['layers'], 
                    hiddens= jcfg['DDPG']['hiddens'],
                    action_dim= a_dim, 
                    state_dim= s_dim, 
                    action_bound= a_bound,
                    REPLACEMENT= jcfg['DDPG']['replacement'],
                    LR_decay= jcfg['DDPG']['LR_decay'],
                    discrete= jcfg['DDPG']['discrete'],
                    LR_A= jcfg['DDPG']['LR_A'],    # learning rate for actor
                    LR_C= jcfg['DDPG']['LR_C'],    # learning rate for critic
                    GAMMA= jcfg['DDPG']['GAMMA'],     # reward discount
                    TAU= jcfg['DDPG']['TAU'],      # soft replacement
                    MEMORY_CAPACITY= jcfg['DDPG']['MEMORY_CAPACITY'],
                    BATCH_SIZE= jcfg['DDPG']['BATCH_SIZE']
        )
    

def ddpg_method_train( env, model,  model_status, env_action, model_reward, check_steps, max_steps, episodes, modelName, show):
    
    print(episodes, max_steps ,show)
    RENDER=False
    #maxReward= model.load( modelName)
    sampling= True
    var=3.
    END_POINT = (200 - 10) * (14/30)
    running_r= None
    maxReward= model.load( modelName)
    for i_episode in range(episodes):
        # s = (hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements.)
        s = env.reset()
        ep_r = 0
        #while True:
        for j in range(max_steps):
            if RENDER:
                env.render()
            if RENDER:
                env.render()
            s = model_status(s)
            a = model.actor.choose_action(s)
            s_, r, done, _ = env.step( env_action( a, running_r))    # r = total 300+ points up to the far end. If the robot falls, it gets -100.
            s_= model_status(s_)
            r = model_reward(r)
            ep_r += r

            if 0 <show:
                transition = np.hstack((s, a, [r], s_))
                max_p = np.max( model.M.tree.tree[-model.M.tree.capacity:])
                model.M.store(max_p, transition)
                model.learn()
                
            if done:
                if running_r== None:
                    running_r = ep_r
                else:
                    running_r = 0.95*running_r + 0.05*ep_r
                if running_r > show: RENDER = True
                #else: RENDER = False
                
                if 0<show and maxReward< running_r:
                    maxReward= running_r
                    model.save( modelName, maxReward)

                done = '| Achieve ' if env.unwrapped.hull.position[0] >= END_POINT else '| -----'
                print('Episode:', i_episode, "steps:", j,
                    done,
                    '| Running_r: %.2f' % int(running_r),
                    '| Epi_r: %.2f' % ep_r,
                    '| Exploration: %.3f' % var,
                    '| Pos: %.i' % int(env.unwrapped.hull.position[0]),
                    
                    )
                break
            
            s = s_
            if 0< show:
                model.sess.run( model.INCREASE_GS)
            if 0<show and running_r!= None and check_steps(max_steps, running_r) < j:
                running_r = 0.95*running_r + 0.05*ep_r
                done = '| Break '
                print('Episode:', i_episode, "steps:", j,
                    done,
                    '| Running_r: %.2f' % int(running_r),
                    '| Epi_r: %.2f' % ep_r,
                    '| Exploration: %.3f' % var,
                    '| Pos: %.i' % int(env.unwrapped.hull.position[0]),
                    
                    )
                break;
            
            #if done== True or j == max_steps-1:
                #print('Episode:', i, ' Reward: %i' % int(ep_reward), "pointer:%d"%model.M.pointer, "Distance:%f"%dist)
                #if model.MEMORY_CAPACITY//20 < model.M.pointer:
                    
                    ##if maxReward < ep_reward:
                        ##maxReward= ep_reward
                        ##model.save(modelName, maxReward)
                        
                    #if  show < ep_reward:
                        #RENDER = True
                    
                #break
        
    #model.save(modelName, maxReward)
