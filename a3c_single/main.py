"""
Faster a3c signle agent for stock scalping trading!!
    - multi threading a3c version
    - update per episode
    - we can give process count as much as server allows
    . Algorithm step
         1. process could gather episode samples and calculate gradient of local network
         2. send its gradient and update global network and pull global network gradient into local network itself

0. algorithms
    . learner is only one and actor could have as many as n_threads configured
    . learner will update target network if total_step_count, global shared variable % update_step_interval  == 0

1. objects
    . agent(env, bsa, boa, ssa, soa)

2. global shared variables between threads
    . total_step_count
        across threads, it represents total step counts
    . total_episode
        across threads, it represents total episodes
    . scores
        across threads, it represents all scores from all episodes over all threads running.

3. threading lock
    N/A

4. Hyper-parameter setting
    referenced by
        https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55
    We start with same hyperparameter with original dqn paper's one
    .   Epsilon decays from 1.0 to 0.02 for the first 100k frames, then epsilon kept 0.02,
    .   Target network synched every 1k frames, (equals to 1000 steps)
    .   Simple replay buffer with size 100k was initially prefetched with 10k transitions before training,
        -> warm_up_steps = 10000 replay_buffer_size = 100,000
    .   Gamma=0.99,
    .   Adam with learning rate 1e-4,\
    Every training step, one transition from the environment is added to the replay buffer
    and training is performed on 32 transitions uniformly sampled from the replay buffer,

"""
from gym_core import tgym
import numpy as np
import random
from keras.models import Model
from keras.layers import Dense, Concatenate
from collections import deque
import glob
import copy
from a3c_agent.loadnetwork import *
import csv
import threading
import tensorflow as tf
from keras import backend as K
import time
from keras.optimizers import Adam

SHARED_GRAPH = tf.get_default_graph()
SESS = tf.Session(graph=SHARED_GRAPH)
SESS = K.get_session()

actor, critic = load_model(g=SHARED_GRAPH)

SESS.run(tf.global_variables_initializer())

N_THREADS = 32
N_MAX_EPISODE = 100
TOTAL_STEP_COUNT = 0
TOTAL_EPISODE = 0
SCORES = []

# Hyperparameter
C_WARM_UP_STEP = 10000
LR = 1E-4
EPSILON_MIN = 0.02
EPSILON_MAX = 1.0
EPSILON_PERIOD_DECAYING = 100000

class DDQNAgent:
    def __init__(self, g, thread_idx, actor, critic):
        self.thread_idx = thread_idx
        self.g_n = g

        self.actor = actor
        self.critic = critic

        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]
        self.epsilon = 1.
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9995
        self.batch_size = 64
        self.train_start = 1000
        self.target_update_interval = 10000

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self.discount_factor = 0.99

    # todo : this function can synchronize local network same as global network
    def update_target_model(self,  global_network, local_network):
        with self.g_n.as_default():
            global_network.set_weights(local_network.get_weights())

    def append_sample(self, state, action, reward, next_state, done):
        self.states.append(state)
        act = np.zeros(2)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def discount_rewards(self, done=True):
        discounted_rewards = np.zeros_like(self.rewards)
        running_add = 0
        if not done:
            pass
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * 0.999 + self.rewards[t]  # todo : a3c discount_factor = 0.999
            discounted_rewards[t] = running_add
        return discounted_rewards

    # update policy network and value network every episode
    def train_episode(self, done):
        discounted_rewards = self.discount_rewards(done)

        inputs = []
        with self.g_n.as_default():
            for d_i in range(self.data_num):
                inputs.append(np.asarray([state[d_i] for state in self.states]))
            values = self.critic.predict(inputs)
        values = np.reshape(values, len(values))

        advantages = discounted_rewards - values

        actor_op_inputs = []
        for i in range(self.data_num):
            actor_op_inputs.append(inputs[i])
        actor_op_inputs.append(self.actions)
        actor_op_inputs.append(advantages)

        critic_op_inputs = []
        for i in range(self.data_num):
            critic_op_inputs.append(inputs[i])
        critic_op_inputs.append(discounted_rewards)

        self.optimizer[0](actor_op_inputs)
        self.optimizer[1](critic_op_inputs)
        self.states, self.actions, self.rewards = [], [], []

    def get_action(self, state):
        inputs = []
        for i in range(self.data_num):
            inputs.append(state[i].reshape((1,) + state[i].shape))
        with self.g_n.as_default():
            policy = self.actor.predict(inputs)[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def actor_optimizer(self):
        action = K.placeholder(shape=(None, 2))
        advantages = K.placeholder(shape=(None, ))

        policy = self.actor.output
        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)

        loss = -K.sum(eligibility)
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        actor_loss = loss + 0.01*entropy

        optimizer = Adam(lr=1e-3) # todo : a3c lr = 1e-3
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)

        inputs = []
        for i in range(2):
            inputs.append(self.actor.input[i])
        inputs.append(action)
        inputs.append(advantages)

        train = K.function(inputs, [], updates=updates)
        return train

    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None, ))
        value = self.critic.output
        loss = K.mean(K.square(discounted_reward - value))
        optimizer = Adam(lr=1e-3) # todo : a3c lr = 1e-3
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)

        inputs = []
        for i in range(2):
            inputs.append(self.critic.input[i])
        inputs.append(discounted_reward)

        train = K.function(inputs, [], updates=updates)
        return train


class Agents(threading.Thread):
    additional_reward_rate = 0.1

    def __init__(self, idx, n_max_episode, file_dir):
        super(Agents, self).__init__()
        self.env = MyTGym(episode_type='0', percent_goal_profit=2, percent_stop_loss=5, episode_duration_min=63)
        self.n_max_episode = n_max_episode
        self.train_log_dir = file_dir
        self.idx = idx

        global SHARED_GRAPH, actor, critic

        self.agent = DDQNAgent(SHARED_GRAPH, idx, actor, critic)

        self.sequence = 0

        self.sample_buffer = list()
        self.remain_step = 0
        self.trainable = False

        self.buy_price = 0
        self.sell_price = 0
        self.thread_step = 0

    def get_action(self, state):
        return self.agent.get_action(state)

    def append_sample(self, state, action, reward, next_state, done):
        self.agents.append_sample(state, action, reward, next_state, done)

    def _append_buffer_sample(self):
        self.sample_buffer[0][2] += self.sample_buffer[1][2] * self.additional_reward_rate + self.sample_buffer[3][2] * 0.5
        self.sample_buffer[1][2] += self.sample_buffer[3][2] * 0.5
        self.sample_buffer[2][2] += self.sample_buffer[3][2] * 0.5
        sb = self.sample_buffer

        for idx, agent in enumerate(self.agents):
            agent.append_sample(sb[idx][0], sb[idx][1], sb[idx][2], sb[idx][3], sb[idx][4])

        return [self.sample_buffer[0][2], self.sample_buffer[1][2], self.sample_buffer[2][2], self.sample_buffer[3][2]]

    def train_agents(self):
        for agent in self.agents:
            agent.train_model()

    # Thread interactive with environment
    def run(self):
        global TOTAL_EPISODE
        global TOTAL_STEP_COUNT
        global SHARED_GRAPH

        for ep in range(self.n_max_episode):
            try:
                print('{} thread {} episode started'.format(self.idx, ep))
                episode_start_time = time.time()
                done = False
                state = self.env.reset()

                buy_count = 0
                profit = 0
                profit_comm = 0
                buy_price, sell_price = 0, 0
                commission = 0.33
                steps = 0
                reward_sum = 0
                while not done:

                    action = self.get_action(state)

                    next_state, reward, done, info = self.env.step(action)

                    if action == 1:  # buy
                        if buy_price != 0:
                            reward = -10
                        else:
                            buy_count += 1
                            buy_price = self.env.holder_observation[-2][0] + 100

                    if action == 2: # sell
                        sell_price = self.env.holder_observation[-2][0] + 100
                        if buy_price != 0: # if it didn't have not bought yet.
                            profit += (sell_price - buy_price) / buy_price
                            profit_comm += (sell_price - buy_price) / buy_price - commission * 0.01
                            reward = sell_price - buy_price - 0.33
                            buy_price = 0
                        else:
                            reward = -10

                    # not using _reward() in gym, just calculate reward right here.
                    reward_sum += reward

                    steps = steps + 1

                    self.append_sample(state, action, reward, next_state, done)

                    TOTAL_STEP_COUNT += 1
                    state = next_state

                    if steps >= 1 * 60 * 60 - 50*60 :  # todo : -3000 for test
                        done = True
                        TOTAL_EPISODE = TOTAL_EPISODE + 1

                        try:
                            agent.train_episode(True)
                        except Exception as e:
                            print('got error {}'.format(e))
                            pass

                profit = round(profit * 100, 5)
                profit_comm = round(profit_comm * 100, 5)
                if buy_count == 0:
                    avg_profit, avg_profit_comm = 0, 0
                else:
                    avg_profit = round(profit / buy_count, 5)
                    avg_profit_comm = round(profit_comm / buy_count, 5)

                print('ep :', ep, end='  ')
                print('epsilon :', round(self.agents[0].epsilon, 3), end='  ')
                print('buy :', buy_count)
                print('reward sum :', round(reward_sum, 5), end='  ')
                print('profit sum :', profit, end='  ')
                print('profit avg :', avg_profit, end='  ')
                print('profit(commission) sum :', profit_comm, end='  ')
                print('profit(commission) avg :', avg_profit_comm, end='  ')
                print('ep time :', int(time.time() - episode_start_time))

                # todo : it could occur resource deadlock, so that it could be reason being slow.. but for now, move on!
                train_log_file = open(self.train_log_dir, 'a', encoding='utf-8', newline='')
                train_log_writer = csv.writer(train_log_file)
                train_log_writer.writerow([ep, reward_sum, profit, avg_profit, profit_comm, avg_profit_comm])
                train_log_file.close()

            except:
                pass

class MyTGym(tgym.TradingGymEnv):
    # data shape
    rows = 10
    columns = 2
    seconds = 120
    channels = 2
    features = 11

    holder_observation = deque(np.array([[0 for x in range(52)] for y in range(seconds)]), maxlen=seconds)

    def _rewards(self, observation, action, done, info):
        return None

    def observation_processor(self, observation):
        self.holder_observation.append(observation)

        x1 = np.zeros([self.rows, self.columns, self.seconds, self.channels])
        x2 = np.zeros([self.seconds, self.features])

        for row in range(self.rows):
            for column in range(self.columns):
                for second in range(self.seconds):
                    for channel in range(self.channels):
                        x1[row][column][second][channel] = self.holder_observation[second][11 + channel*20 + column*10 + row]

        for second in range(self.seconds):
            for feature in range(self.features):
                x2[second][feature] = self.holder_observation[second][feature]

        return [x1, x2]


if __name__ == '__main__':
    train_log_file_dir = 'train_log.csv'
    agents = []

    for i in range(N_THREADS):
        agent = Agents(i, N_MAX_EPISODE, train_log_file_dir)
        agents.append(agent)

    import time
    for a in agents:
        time.sleep(1)
        a.start()