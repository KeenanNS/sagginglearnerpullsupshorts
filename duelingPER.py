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

MIN_REPLAY_MEM = 150
GAMMA = 0.995
EPISODES = 500
lr = 0.01
AGGREGATE_STATS_EVERY = 10
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.005
epsilon = .995
MIN_REWARD = 100
fit_count = 0
MODEL_NAME = 'duelingDDQN'
if not os.path.isdir('models'):
    os.makedirs('models')

env = gym.make('LunarLander-v2')
num_actions = env.action_space.n
obs_dims = env.observation_space.shape[0]

def huber_loss(loss):
    return 0.5 * loss ** 2 if abs(loss) < 1.0 else abs(loss) - 0.5

class DDDQN(keras.Model):
    def __init__(self, obs_dims, num_actions):
        super(DDDQN, self).__init__()
        self.dense1 = layers.Dense(64, activation = 'relu', kernel_initializer = keras.initializers.he_normal())
        self.dense2 = layers.Dense(64, activation = 'relu', kernel_initializer = keras.initializers.he_normal())
        self.dense3 = layers.Dense(64, activation = 'relu', kernel_initializer = keras.initializers.he_normal())
        self.adv_dense = layers.Dense(64, activation = 'relu', kernel_initializer = keras.initializers.he_normal())
        self.adv_out = layers.Dense(num_actions, activation = 'relu', kernel_initializer = keras.initializers.he_normal())


        self.v_dense = layers.Dense(64, activation = 'relu', kernel_initializer = keras.initializers.he_normal())
        self.v_out = layers.Dense(1, kernel_initializer = keras.initializers.he_normal())
        self.lambda_layer = layers.Lambda(lambda x: x - tf.reduce_mean(x))
        self.combine = layers.Add()

    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.dense3(x)
        adv = self.adv_dense(x)
        adv = self.adv_out(adv)

        v = self.v_dense(x)
        v = self.v_out(v)
        norm_adv = self.lambda_layer(adv)
        combined = self.combine([v, norm_adv])
        print('combined : ', combined, 'norm adv : ', norm_adv)
        return combined


model = DDDQN(obs_dims, num_actions)
model.compile(epochs = 1, lr= lr, optimizer = 'Adam', loss = 'mse')
target_model = DDDQN(obs_dims, num_actions)

for t, e in zip (target_model.get_weights(), model.get_weights()):
    t.assign(e)

def update_target(model, target_model):
    for t, e in zip (target_model.get_weights(), model.get_weights()):
        t.assign(t * (1 - TAU) + e * TAU)



class Agent:
    def __init__(self, lr, GAMMA, obs_dims, num_actions):
        self.model = DDDQN(obs_dims, num_actions)
        self.model.compile(epochs = 1, lr= lr, optimizer = 'Adam', loss = 'mse')
        self.target_model = DDDQN(obs_dims, num_actions)
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=10000)
        self.target_update_counter = 0
        self.lr = lr
        self.GAMMA = GAMMA
        self.num_actions = num_actions
        self.obs_dims = obs_dims


    def update_replay_memory(self, obs):
        self.replay_memory.append(obs)

    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, obs_dims))

    def train(self, done, step):
        if len(self.replay_memory)< MIN_REPLAY_MEM:
            return

        minibatch = random.sample(self.replay_memory, 64)

        # current_states = np.array([obs[0] for obs in minibatch])
        # current_qs_list = self.model.predict(current_states.reshape(-1, obs_dims))
        #
        # new_current_states = np.array([obs[3] for obs in minibatch])
        # future_qs_list = self.model.predict(new_current_states.reshape(-1, obs_dims))
        #
        # target_future_qs = self.target_model.predict(new_current_states.reshape(-1, obs_dims))
        current_qs = self.model.predict(state)
        next_qs = self.model.predict(next_state)
        target_qs = self.target_model.predict(next_state)
        states = []

        # X = []
        # y = []

        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            states.append(state)
            if not done:



                # the current Q network is going to choose the action while later
                # the target mdoel will evaluate it

                a = np.argmax(target_q)
                new_qs[idx][action] = reward + self.GAMMA * target_qs[idx][a]
            else:
                new_qs[idx][action] = reward

            #current_qs = current_qs_list[idx]
            #current_qs[action] = new_q

            #X.append(current_states[idx])
            #y.append(current_qs)
        self.model.fit(np.array(states), np.array(new_qs), batch_size = len(minibatch), verbose = 0, shuffle = False)


        if done:
            self.target_update_counter += 1
        if self.target_update_counter > 5:
            print('updating target model')
            update_target(model, target_model)
            self.target_update_counter = 0


agent = Agent(lr, GAMMA, num_actions= num_actions, obs_dims= obs_dims)
ep_rewards = []


for episode in range (1, EPISODES):
    ep_reward = 0
    step = 1
    current_state = env.reset()
    done = False
    while not done:
        if episode % 5 == 0 :
            env.render()
        if epsilon > np.random.random():
            action = np.random.randint(0, num_actions)
        else:
            action = np.argmax(agent.get_qs(current_state.reshape(-1, obs_dims)))

        new_state, reward, done, _ = env.step(action)

        ep_reward+= reward

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        if len(agent.replay_memory) > 132:
            agent.train(done, step)
        current_state = new_state
        step += 1


    ep_rewards.append(ep_reward)
    avg_rew = sum(ep_rewards[-50:])/50


    print('episode: ', episode, 'reward: ', ep_reward, 'avg_reward: ', avg_rew)
    print('------------------------------------------------------------------')


    if len(ep_rewards)>20:
        avg_rews = sum(ep_rewards[-20:])/20

        if avg_rews > 200:
            print('saving model')
            agent.model.save(f'models/{MODEL_NAME}__reward__{avg_rew}__episode__{episode}___at__{int(time.time())}.model')
            sys.exit('acheived goal')
    ep_rewards.append(ep_reward)

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
