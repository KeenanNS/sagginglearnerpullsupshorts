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
import priority_experience_replay

MIN_REPLAY_MEM = 1000
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

env = gym.make('LunarLander-v2')
#env.unwrapped
num_actions = env.action_space.n
obs_dims = env.observation_space.shape[0]
obs_shape = (obs_dims,)
class SumTree(object):
    data_pointer = 0

    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)


    # Here we define function that will add our priority score in the sumtree leaf and add the experience in data:
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update (tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0

    # Update the leaf priority score and propagate the change through tree
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop in the reference code
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    # Here build a function to get a leaf from our tree. So we'll build a function to get the leaf_index, priority value of that leaf and experience associated with that leaf index:
    def get_leaf(self, v):
        parent_index = 0

        # the while loop is faster than the method in the reference code
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else: # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node

# Now we finished constructing our SumTree object, next we'll build a memory object.
class Memory(object):  # stored as ( state, action, reward, next_state ) in SumTree
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        self.tree = SumTree(capacity)

    # Next, we define a function to store a new experience in our tree.
    # Each new experience will have a score of max_prority (it will be then improved when we use this exp to train our DDQN).
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)   # set the max priority for new priority

    # Now we create sample function, which will be used to pick batch from our tree memory, which will be used to train our model.
    # - First, we sample a minibatch of n size, the range [0, priority_total] into priority ranges.
    # - Then a value is uniformly sampled from each range.
    # - Then we search in the sumtree, for the experience where priority score correspond to sample values are retrieved from.
    def sample(self, n):
        # Create a minibatch array that will contains the minibatch
        minibatch = []

        b_idx = np.empty((n,), dtype=np.int32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment

        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            b_idx[i]= index

            minibatch.append([data[0],data[1],data[2],data[3],data[4]])

        return b_idx, minibatch

    # Update the priorities on the tree
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)



class Agent:
    def __init__(self, lr, GAMMA, obs_dims, num_actions, TAU):
        self.lr = lr
        self.num_actions = num_actions
        self.obs_dims = obs_dims
        self.model = self.ddqn(obs_dims = self.obs_dims, num_actions = self.num_actions, lr = self.lr)
        self.target_model = self.ddqn(obs_dims = self.obs_dims, num_actions = self.num_actions, lr = self.lr)
        self.target_update_counter = 0
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.MEMORY = Memory(10000)

        # self.Save_Path = 'Models'
        # if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        # self.scores, self.episodes, self.average = [], [], []
        #
        # self.Model_name = os.path.join(self.Save_Path, 'lunarPER'+"_e_greedy.h5")


    def ddqn(self, obs_dims, num_actions, lr):
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

    def update_replay_memory(self, obs):
        self.MEMORY.store(obs)

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
        # if len(self.MEMORY)< MIN_REPLAY_MEM:
        #     return

        tree_idx, minibatch = self.MEMORY.sample(64)
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
        target_old = np.array(target)
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

            indices = np.arange(64, dtype=np.int32)
            absolute_errors = np.abs(target_old[indices, np.array(action)]-target[indices, np.array(action)])
            # Update priority
            self.MEMORY.batch_update(tree_idx, absolute_errors)

        self.model.fit(state, target, batch_size=64, verbose=0)

agent = Agent(lr, GAMMA, num_actions= num_actions, obs_dims= obs_dims, TAU=TAU)
ep_rewards = []
avg_rews = 0
for episode in range (1, EPISODES):

    ep_reward = 0
    step = 1
    current_state = env.reset()
    done = False
    while not done:
        if episode %10 == 0:
            env.render()
        if avg_rews > 200:
            env.render()

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
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
    ep_rewards.append(ep_reward)
    avg_rew = sum(ep_rewards)/episode+1



    print('episode: ', episode,'eps', epsilon, 'reward: ', ep_reward, 'avg_recent_reward: ', avg_rews)
    print('------------------------------------------------------------------')


    if len(ep_rewards)>10:
        avg_rews = sum(ep_rewards[-10:])/10
        # agent.model.save('ddqn w/ PER')
        # sys.exit()

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
