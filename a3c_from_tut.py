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
from utils import push_and_pull, record
import multiprocessing as mp
import os
os.environ["OMP_NUM_THREADS"] = "1"


env = gym.make('LunarLander-v2')
env.unwrapped
num_actions = env.action_space.n
obs_dims = env.observation_space.shape[0]
obs_shape = (obs_dims,)
lr = 0.00001
lrb = 5 * lr
gamma = 0.999
EPISODES = 5000;

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
    input = layers.Input(shape = (obs_dims,))
    delta = layers.Input(shape = [1])
    dense1 = layers.Dense(512, activation = 'relu')(input)
    dense2 = layers.Dense(512, activation = 'relu')(dense1)
    dense2 = layers.Dense(132, activation = 'relu')(dense2)
    dense2 = layers.Dense(64, activation = 'relu')(dense2)
    probs = layers.Dense(num_actions, activation = 'softmax')(dense2)
    values = layers.Dense(1, activation = 'linear')(dense2)

    def loss(y_true, y_pred):
        w = K.clip(y_pred, 1e-7, 1-1e-7)
        x = y_true * K.log(w)
        return K.sum(-x * delta)

    actor = Model(input = [input, delta], output = [probs])
    actor.compile(optimizer = optimizers.Adam(lr = lr), loss = loss)
    actor.summary()
    critic = Model(input = [input], output = [values])
    critic.compile(optimizer = optimizers.Adam(lr = lrb), loss = 'mse')
    critic.summary()
    policy = Model(input = [input], output = [probs])
    policy.summary()
    return actor, critic, policy

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, res_queue, name):
        super(Worker, self).__init__()
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_reward, res_queue
        self.gnet = gnet
        self.lnet = build_model()           # local network
        self.env = gym.make('CartPole-v0').unwrapped

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

    def choose_action(self, state):
        state = state[np.newaxis, :]
        probability = self.policy.predict(state)[0]
        #probability = np.nan_to_num(probability)
        action = np.random.choice(num_actions, p= probability)
        return action

    def run(self):
        total_step = 1
        while self.g_ep.value < EPISODES:
            state = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                ep_reward += reward
                state = next_state
                self.train
                buffer_a.append(action)
                buffer_s.append(state)
                buffer_r.append(reward)

                if total_step % UPDATE_FREQ == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = build_model()        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    #opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_reward, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, global_ep, global_ep_reward, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        reward = res_queue.get()
        if r is not None:
            res.append(reward)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
