import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from KukaGymEnv import KukaGymEnv
import tensorflow.compat.v1 as tf
from DDQN import DoubleDQN

MAX_EPISODES = 3000
ON_TRAIN = True
plotTrainHyper=False

with tf.variable_scope('DDQN'):
     q_double=DoubleDQN(
                         double_q=True,
                         n_actions=4, 
                         n_features=3,
                         learning_rate=0.001,
                         reward_decay=0.9,
                         e_greedy=0.95,
                         replace_target_iter=900,
                         memory_size=60000,
                         )

with tf.variable_scope('DQN2'):
     q_double2=DoubleDQN(
                         double_q=False,
                         n_actions=4, 
                         n_features=3,
                         learning_rate=0.001,
                         reward_decay=0.9,
                         e_greedy=0.95,
                         replace_target_iter=900,
                         memory_size=60000,
                         )
def train(RL,env, num_episodes=MAX_EPISODES):
    
    cost_his = []
    step_es=[0]
    total_steps = 0
    episode_rewards = [0.0]
    record=0
    
    for episode in range(num_episodes):
        step_per_es=0
        obs, done = env.reset(), False
        while not done:
            # env.render()
            action=RL.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            step_reward=reward
            RL.store_transition(obs, action, step_reward, obs_)
            if (total_steps > 600):
                RL.learn()
            if done:
                print("num_epidodes:",episode,"episode_reward:",reward)
                break
            step_per_es+=1
            total_steps+=1
            obs=obs_
        step_es[-1]=step_per_es
        episode_rewards[-1] += reward
        episode_rewards.append(0.0)
        step_es.append(0)
    print("training over")
    #RL.save()
    RL.sess.close()
    return episode_rewards, RL.q,step_es

def plotResult_DDQN(env, MAX_EPISODES,q_double):
    episode_rewards_double,q_double_value,total_steps =train(q_double,env)
    rewards_double=episode_rewards_double
    step=total_steps[0:MAX_EPISODES]
    for episode_num_double in range(MAX_EPISODES-101):
        rewards_double[episode_num_double]=sum(rewards_double[episode_num_double:episode_num_double+100])/100
    rewards_double=rewards_double[0:MAX_EPISODES-101]
    plt.plot(np.array(rewards_double), c='b', label='double_deep_q')
    plt.xlabel('episode') 
    plt.ylabel('reward')
    plt.show()
    plt.plot(np.array(q_double_value), c='b', label='double')
    plt.xlabel('training_steps') 
    plt.ylabel('q_eval')
    plt.show()
    plt.plot(np.array(step), c='b', label='steps')
    plt.xlabel('espisode') 
    plt.ylabel('step')
    plt.show()
def plotTrainHyper(env,MAX_EPISODES,q_natural1,q_natural2):
        episode_rewards_natural1,q_natural1_value,step =train(q_natural1,env)
        episode_rewards_natural2,q_natural2_value,step =train(q_natural2,env)
        rewards_natural1=episode_rewards_natural1
        rewards_natural2=episode_rewards_natural2
       
        rewards_natural=[rewards_natural1,rewards_natural2]
        for num_freq in range(2):
            for episode_num_natural in range(MAX_EPISODES-101):
                rewards_natural[num_freq][episode_num_natural]=sum(rewards_natural[num_freq][episode_num_natural:episode_num_natural+100])/100
            rewards_natural[num_freq]=rewards_natural[num_freq][0:MAX_EPISODES-101]
        

        plt.plot(np.array(rewards_natural[0]), c='r', label='DQN')
        plt.plot(np.array(rewards_natural[1]), c='b', label='DDQN')
        plt.legend(loc='best')
        plt.ylabel('rewards')
        plt.xlabel('episodes')
        plt.grid()
        plt.show()





if __name__ == '__main__':
    if ON_TRAIN:
        env= KukaGymEnv(renders=True, isDiscrete=True)
        plotResult_DDQN(env, MAX_EPISODES,q_double)
    if plotTrainHyper:
        env= KukaGymEnv(renders=True, isDiscrete=True)
        plotTrainHyper(env,MAX_EPISODES,q_double,q_double2)
        