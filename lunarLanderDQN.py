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

MIN_REPLAY_MEM = 25
GAMMA = 0.995
EPISODES = 10000
lr = 0.001
AGGREGATE_STATS_EVERY = 5
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.005
epsilon = .995
MIN_REWARD = 100
fit_count = 0
if not os.path.isdir('models'):
    os.makedirs('models')

env = gym.make('LunarLander-v2')
num_actions = env.action_space.n
obs_dims = env.observation_space.shape[0]

# class ModifiedTensorBoard(TensorBoard):
#
#     # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.step = 1
#         self.writer = tf.summary.create_file_writer(self.log_dir)
#
#     # Overriding this method to stop creating default log writer
#     def set_model(self, model):
#         pass
#
#     # Overrided, saves logs with our step number
#     # (otherwise every .fit() will start writing from 0th step)
#     def on_epoch_end(self, epoch, logs=None):
#         self.update_stats(**logs)
#
#     # Overrided
#     # We train for one batch only, no need to save anything at epoch end
#     def on_batch_end(self, batch, logs=None):
#         pass
#
#     # Overrided, so won't close writer
#     def on_train_end(self, _):
#         pass
#
#     # Custom method for saving own metrics
#     # Creates writer, writes custom metrics and closes writer
#     def update_stats(self, **stats):
#         self._write_logs(stats, self.step)

class Agent:
    def __init__(self, lr, GAMMA, obs_dims, num_actions):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=10000)
        self.target_update_counter = 0
        #self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format('first_model', int(time.time())))
        self.lr = lr
        self.GAMMA = GAMMA
        self.num_actions = num_actions
        self.obs_dims = obs_dims


    def create_model(self):
        model = models.Sequential()
        model.add(layers.Dense(132,input_dim = obs_dims, activation = 'relu'))
        model.add(layers.Dense(64, activation = 'relu'))
        model.add(layers.Dense(64, activation = 'relu'))
        model.add(layers.Dense(32, activation = 'relu'))
        model.add(layers.Dense(num_actions, activation = 'softmax'))
        model.compile(loss = 'mse', metrics = ['accuracy'], optimizer = optimizers.Adam(lr = lr))
        return model

    def update_replay_memory(self, obs):
        self.replay_memory.append(obs)

    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, obs_dims))

    def train(self, done, step):
        if len(self.replay_memory)< MIN_REPLAY_MEM:
            return
        minibatch = random.sample(self.replay_memory, 64)

        current_states = np.array([obs[0] for obs in minibatch])
        current_qs_list = self.model.predict(current_states.reshape(-1, obs_dims))

        new_current_states = np.array([obs[3] for obs in minibatch])
        future_qs_list = self.target_model.predict(new_current_states.reshape(-1, obs_dims))


        X = []
        y = []

        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[idx])
                new_q = reward + self.GAMMA * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[idx]
            current_qs[action] = new_q

            X.append(current_states[idx])
            y.append(current_qs)
            self.model.fit(np.array(X), np.array(y), batch_size = len(minibatch), verbose = 0, shuffle = False)


        if done:
            self.target_update_counter += 1
        if self.target_update_counter > 5:
            print('updating target model')
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0




##episode iteration
agent = Agent(lr, GAMMA, num_actions= num_actions, obs_dims= obs_dims)
ep_rewards = []


for episode in range (1, EPISODES):
    #agent.tensorboard.step = episode
    ep_reward = 0
    step = 1
    current_state = env.reset()
    done = False
    while not done:
        if epsilon > np.random.random():
            action = np.random.randint(0, num_actions)
        else:
            action = np.argmax(agent.get_qs(current_state.reshape(-1, obs_dims)))

        new_state, reward, done, _ = env.step(action)

        #if episode % 5 == 0 | episode ==0:
        #env.render()

        ep_reward+= reward
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        if len(agent.replay_memory) > 132:
            agent.train(done, step)
        current_state = new_state
        step += 1
    print('episode: ', episode, 'reward: ', ep_reward)

    ep_rewards.append(ep_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        #agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
    if ep_reward > 170:
        agent.model.save(f'models/{MODEL_NAME}__reward__{ep_reward}__episode__{episode}___at__{int(time.time())}.model'
    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

x = np.arange(EPISODES-1)
x = x.transpose()
y = np.array(ep_rewards)
plt.plot(x,y)
plt.title('rewards vs episodes')
plt.xlabel('episodes')
plt.ylabel('rewards')
plt.show()
