

import tensorflow as tf
import keras
import gym
import threading
import multiprocessing
from keras import layers
from queue import Queue
import numpy as np
#defines
lr = 0.000025
EPISODES = 3000
UPDATE_FREQ = 5

##setup env
#https://medium.com/tensorflow/deep-reinforcement-learning-playing-cartpole-through-asynchronous-advantage-actor-critic-a3c-7eab2eea5296
#https://github.com/smtkymt/A3C/blob/master/a3c/__init__.py
#the network
env = gym.make("LunarLander-v2")
num_actions = env.action_space.n
obs_dims = env.observation_space.shape[0]

class Model(keras.Model):
    def __init__(self, obs_dims, num_actions):
        super(Model, self).__init__()
        self.num_actions = num_actions
        self.obs_dims = obs_dims

        self.dense1 = layers.Dense(512, activation = 'relu')
        self.dense2 = layers.Dense(512, activation = 'relu')
        self.dense3 = layers.Dense(512, activation = 'relu')
        self.dense4 = layers.Dense(512, activation = 'relu')
        self.policy_logits = layers.Dense(self.num_actions, activation = 'softmax')
        self.values = layers.Dense(1, activation = 'linear')

        def call(self, inputs):
    # Forward pass
            x = self.dense1(inputs)
            x = self.dense2(x)
            logits = self.policy_logits(x)
            v1 = self.dense3(inputs)
            v1 = self.dense4(v1)
            values = self.values(v1)
            return logits, values
class Master():
    def __init__(self, obs_dims, num_actions, lr):
        self.obs_dims = obs_dims
        self.num_actions = num_actions
        self.lr = lr
        self.global_model = Model(self.obs_dims, self.num_actions)
        self.opt = tf.compat.v1.train.AdamOptimizer(self.lr, use_locking=True)
        self.global_model(tf.convert_to_tensor(np.random.random((1, self.obs_dims)), dtype=tf.float32))

    def train(self):
        res_queue = Queue()

        workers = [Worker(self.obs_dims,
                          self.num_actions,
                          self.global_model,
                          self.opt,
                          res_queue, i) for i in range(multiprocessing.cpu_count())]

        for i, worker in enumerate(workers):
            print("starting {}th worker".format(i))
            worker.start()

        avg_rews = []
        while True:
            reward = res_queue.get()
            if reward is not None:
                avg_rews.append(reward)
            else:
                break
        [w.join() for w in workers]

        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(os.path.join(self.save_dir,'{} Moving Average.png'.format(self.game_name)))
        plt.show()

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

class Worker(threading.Thread):
    global_episode = 0
    global_avg_reward = 0
    best = 0
    save_lock = threading.Lock()

    def __init__(self,
                 obs_dims,
                 num_actions,
                 global_model,
                 res_queue,
                 opt,
                 idx):
        super(Worker, self).__init__()
        self.obs_dims = obs_dims
        self.num_actions = num_actions
        self.global_model = global_model
        self.res_queue = res_queue
        self.worker_idx = idx
        self.local_model = Model(self.obs_dims, self.num_actions)
        self.loss = 0.0

    def run(self):
        tot_steps = 1
        mem = Memory()
        while Worker.global_episode < EPISODES:
            current_state = env.reset()
            mem.clear
            ep_reward = 0
            step = 0
            self.loss = 0
            done = False
            while not done:
                logits, _ = self.local_model()
                tf.convert_to_tensor(current_state[None, :], dtype=tf.float32)
                probs = tf.nn.softmax(logits)

                action = np.random.choise(self.num_actions, p=probs.numpy()[0])
                new_state, reward, done, _ = env.step(action)
                ep_reward += rewards
                mem.store(state, action, reward)
                step += 1

                if step % UPDATE_FREQ == 0:

                    with tf.GradientTape() as tape:
                        tot_loss = self.compute_loss(done, new_state, mem, GAMMA)

                    self.loss += tot_loss
                    #calculating the local gradients of the network
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    #push local gradients onto the global model
                    self.opt.apply_gradients(zip(grads, self.global_model.trainable_weights))

                    mem.clear()

                    if done:
                        Worker.global_avg_reward = record(Worker.global_episode, ep_reward, self.worker_idx, Worker.global_avg_reward, self.res_queue, self.loss, self.step)
                        Worker.global_episode += 1
                        print("episode : ", Worker.global_episode, "reward : ", ep_reward, "global average: ", Worker.global_avg_reward)

                    step +=1
                    current_state = new_state
                    tot_steps +=1
                self.res_queue.put(None)

    def compute_loss(self, done, new_state, mem, gamma=0.999):
        if done:
            reward_sum = 0
        else:
            reward_sum = self.local_model(tf.convert_to_tensor(new_state[None, :], dtype=tf.float32))[-1].numpy()[0]

        discounted_rewards = []

        for reward in mem.rewards[::-1]:
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.local_model(tf.convert_to_tensor(np.vstack(mem.states), dtype= tf.float32))

        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype = tf.float32) - values

        value_loss = advantage ** 2

        actions = tf.one_hot(mem.actions, self.num_actions, dtype = tf.float32)

        policy = tf.nn.softmax_cross_entropy_logits_v2(labels = actions, logits = logits)
        entropy = tf.reduce_sum(policy * tf.log(policy + 1e-20), axis = 1)


        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        tot_loss = tf.reduce_mean(0.5 * value_loss + policy_loss)

        return tot_loss

agent = Master(obs_dims, num_actions, lr)

###################################

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
import threading
import multiprocessing

env = gym.make('LunarLander-v2')
env.unwrapped
num_actions = env.action_space.n
obs_dims = env.observation_space.shape[0]
obs_shape = (obs_dims,)
lr = 0.000025
lrb = 5 * lr
gamma = 0.999
EPISODES = 2500;


class workers:
    workers = []
    def __init__(self):
        for i in range (4):
            workers.append(Agent(lr, gamma, obs_dims, num_actions, lrb))
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

class Agent:
    def __init__(self, lr, GAMMA, obs_dims, num_actions, lrb):
        Thread.__init__(self)
        self.lock = Lock()
        self.lr = lr
        self.num_actions = num_actions
        self.obs_dims = obs_dims
        self.gamma = gamma
        self.num_actions = num_actions
        self.obs_dims = obs_dims
        self.lrb = lrb
        self.actor, self.critic, self.policy = self.build_model()
        self.global_model = self.build_model()


    def build_model(self):
        input = layers.Input(shape = (self.obs_dims,))
        delta = layers.Input(shape = [1])
        dense1 = layers.Dense(512, activation = 'relu', initializer = initializers.he_normal())(input)
        dense2 = layers.Dense(512, activation = 'relu', initializer = initializers.he_normal())(dense1)
        dense2 = layers.Dense(512, activation = 'relu', initializer = initializers.he_normal())(dense2)
        probs = layers.Dense(self.num_actions, activation = 'softmax')(dense2)
        values = layers.Dense(1, activation = 'linear')(dense2)

        def loss(y_true, y_pred):
            w = K.clip(y_pred, 1e-8, 1-1e-8)
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



Agent = workers
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
        action = Agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        ep_reward+= reward
        state = next_state
        Agent.train(state, action, reward, next_state, done)

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
