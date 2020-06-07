import tensorflow as tf
import keras
from keras import backend as K
from keras import layers, models, optimizers
import numpy as np
import gym
from keras.callbacks import TensorBoard
from collections import deque
import time
import os
import random
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model, load_model

env = gym.make('LunarLander-v2')
env.unwrapped
num_actions = env.action_space.n
obs_dims = env.observation_space.shape[0]
obs_shape = (obs_dims,)
lr = 0.00001
lrb = 5 * lr
gamma = 0.999
EPISODES = 2500;

class Agent:
    def __init__(self, lr, GAMMA, obs_dims, num_actions, lrb):
        self.lr = lr
        self.num_actions = num_actions
        self.obs_dims = obs_dims
        self.gamma = gamma
        self.num_actions = num_actions
        self.obs_dims = obs_dims
        self.lrb = lrb
        self.actor, self.critic, self.policy = self.build_model()


    def build_model(self):
        input = layers.Input(shape = (self.obs_dims,))
        delta = layers.Input(shape = [1])
        dense1 = layers.Dense(512, activation = 'relu')(input)
        dense2 = layers.Dense(512, activation = 'relu')(dense1)
        dense2 = layers.Dense(512, activation = 'relu')(dense2)
        probs = layers.Dense(self.num_actions, activation = 'softmax')(dense2)
        values = layers.Dense(1, activation = 'linear')(dense2)

        def loss(y_true, y_pred):
            w = K.clip(y_pred, 1e-7, 1-1e-7)
            x = y_true * K.log(w)
            return K.sum(-x * delta)

        actor = Model(input = [input, delta], output = [probs])
        actor.compile(optimizer = optimizers.Adam(lr = self.lr), loss = loss)
        actor.summary()
        critic = Model(input = [input], output = [values])
        critic.compile(optimizer = optimizers.Adam(lr = self.lrb), loss = 'mse')
        critic.summary()
        policy = Model(input = [input], output = [probs])
        policy.summary()
        return actor, critic, policy


    def choose_action(self, state):
        state = state[np.newaxis, :]
        probability = self.policy.predict(state)[0]
        #probability = np.nan_to_num(probability)
        action = np.random.choice(num_actions, p= probability)
        return action

    def train(self, state, action, reward, next_state, done):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]
        critic_val_ = self.critic.predict(next_state)
        critic_val = self.critic.predict(state)

        target = reward + self.gamma * critic_val_ * (1- int(done))
        delta = target - critic_val

        actions = np.zeros([1, self.num_actions])
        actions[np.arange(1), action] = 1

        self.actor.fit([state, delta], actions, verbose = 0)
        self.critic.fit(state, target, verbose = 0)



agent = Agent(lr, gamma, obs_dims, num_actions, lrb)
ep_rewards = []
avg_rews = 0
for episode in range (1, EPISODES):
    # if avg_rews >-200:
    #     env = gym.wrappers.Monitor(env, 'C:Users/ksimps57/python/', video_callable=lambda episode_id: episode_id==EPISODES)

    ep_reward = 0
    step = 1
    state = env.reset()
    done = False
    while not done:
        if episode %25 == 0:
            env.render()
        # if avg_rews > 200:
        #     env.render()
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        ep_reward+= reward
        state = next_state
        agent.train(state, action, reward, next_state, done)

        step += 1

    ep_rewards.append(ep_reward)
    avg_rew = sum(ep_rewards)/episode+1

    print('episode: ', episode,'eps', ep_reward, 'avg_recent_reward: ', avg_rews)
    print('------------------------------------------------------------------')

    if len(ep_rewards)>10:
        avg_rews = sum(ep_rewards[-10:])/10

        if avg_rews > 220:
    #agent.model.save_weights(f'model_weights/{MODEL_NAME}__reward__{avg_rew}__episode__{episode}___at__{int(time.time())}.h5')
            make_graph(EPISODES, ep_rewards)
            sys.exit('acheived goal')

def make_graph(EPISODES, ep_rewards):        #agent.model.save_weights(f'models_weights/{MODEL_NAME}__reward__{avg_rews}__episode__{episode}___at__{int(time.time())}.h5')
    x = np.arange(EPISODES-1)
    x = x.transpose()
    y = np.array(ep_rewards)
    plt.plot(x,y)
    plt.title('rewards vs episodes')
    plt.xlabel('episodes')
    plt.ylabel('rewards')
    plt.show()

make_graph(EPISODES, ep_rewards)
