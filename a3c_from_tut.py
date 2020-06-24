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
