import tensorflow as tf
import keras
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

MIN_REPLAY_MEM = 2000
GAMMA = 0.999
EPISODES = 2000
lr = 0.0004
AGGREGATE_STATS_EVERY = 10
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.995
epsilon = 1.0
MIN_REWARD = 100
fit_count = 0
MODEL_NAME = 'duelingDDQN'
TAU = 0.1
if not os.path.isdir('models'):
    os.makedirs('models')

env = gym.make('LunarLander-v2')
env.unwrapped
num_actions = env.action_space.n
obs_dims = env.observation_space.shape[0]
obs_shape = (obs_dims,)

def ddqn(obs_dims, num_actions, lr):
    X_input = layers.Input(shape =(obs_dims,))
    X = X_input
    X = layers.Dense(64, input_shape = (obs_shape), activation = 'relu', kernel_initializer = keras.initializers.he_normal())(X)
    X = layers.Dense(32, activation = 'relu', kernel_initializer = keras.initializers.he_normal())(X)
    X = layers.Dense(32, activation = 'relu', kernel_initializer = keras.initializers.he_normal())(X)
    #X = layers.Dense(16, activation = 'relu', kernel_initializer = keras.initializers.he_normal())(X)
    adv = layers.Dense(num_actions, activation = 'relu', kernel_initializer = keras.initializers.he_normal())(X)
    adv = action_advantage = layers.Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(num_actions,))(adv)
    V = layers.Dense(1, activation = 'relu', kernel_initializer = keras.initializers.he_normal())(X)
    V = state_value = layers.Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(num_actions,))(V)
    X = layers.Add()([adv, V])
    model = Model(inputs = X_input, outputs = X )
    model.compile(loss = 'mse', optimizer = optimizers.Adam(lr = lr), metrics = ['accuracy'])
    model.summary()
    return model


class Agent:
    def __init__(self, lr, GAMMA, obs_dims, num_actions, TAU = TAU):
        self.lr = lr
        self.num_actions = num_actions
        self.obs_dims = obs_dims
        self.model = ddqn(obs_dims = self.obs_dims, num_actions = self.num_actions, lr = self.lr)
        self.target_model = ddqn(obs_dims = self.obs_dims, num_actions = self.num_actions, lr = self.lr)
        self.replay_memory = deque(maxlen=120000)
        self.target_update_counter = 0
        self.GAMMA = GAMMA
        self.TAU = TAU


    def update_replay_memory(self, obs):
        self.replay_memory.append(obs)

    def update_target(self, model, target_model):
        q_model_theta = self.model.get_weights()
        target_model_theta = self.target_model.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_model_theta, target_model_theta):
            target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
            target_model_theta[counter] = target_weight
            counter += 1
        self.target_model.set_weights(target_model_theta)


    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, obs_dims))

    def train(self, done, step):
        if len(self.replay_memory)< MIN_REPLAY_MEM:
            return

        minibatch = random.sample(self.replay_memory, 132)
        batch_size = len(minibatch)

        state = np.zeros((batch_size, self.obs_dims))
        next_state = np.zeros((batch_size, self.obs_dims))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        # predict Q-values for starting state using the main network

        target = self.model.predict(state.reshape(-1, obs_dims))
        # predict best action in ending state using the main network
        target_next = self.model.predict(next_state.reshape(-1, obs_dims))
        # predict Q-values for ending state using the target network
        target_val = self.target_model.predict(next_state.reshape(-1, obs_dims))

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.GAMMA * (target_val[i][a])

        self.model.fit(state, target, batch_size=132, verbose=0)

agent = Agent(lr, GAMMA, num_actions= num_actions, obs_dims= obs_dims)
ep_rewards = []
avg_rews = 0
for episode in range (1, EPISODES):

    ep_reward = 0
    step = 1
    current_state = env.reset()
    done = False
    while not done:
        # if episode %10 == 0:
        #     env.render()
        # if avg_rews > 200:
        #     env.render()

        if epsilon > np.random.random():
            action = np.random.randint(0, num_actions)
        else:
            action = np.argmax(agent.get_qs(current_state))

        new_state, reward, done, _ = env.step(action)

        ep_reward+= reward


        agent.update_replay_memory((current_state, action, reward, new_state, done))
        #if len(agent.replay_memory) > 132:
        current_state = new_state
        agent.train(done, step)

        step += 1

    agent.update_target(agent.model, agent.target_model)
    if epsilon > MIN_EPSILON:
        if len(agent.replay_memory) > MIN_REPLAY_MEM:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
    ep_rewards.append(ep_reward)
    avg_rew = sum(ep_rewards)/episode+1



    print('episode: ', episode,'eps', epsilon, 'reward: ', ep_reward, 'avg_recent_reward: ', avg_rews)
    print('------------------------------------------------------------------')


    if len(ep_rewards)>10:
        avg_rews = sum(ep_rewards[-10:])/10

        if avg_rews > 250:


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
