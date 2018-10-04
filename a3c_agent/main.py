"""

Faster a3c aggregated agent for stock scalping trading!!
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

SESS = K.get_session()
SESS.run(tf.global_variables_initializer())

N_THREADS = 4
N_MAX_EPISODE = 100
TOTAL_STEP_COUNT = 0
TOTAL_EPISODE = 0
SCORES = []

# Hyperparameter
C_RB_SIZE = 100000
C_WARM_UP_STEP = 10000
C_SYNC_STEP_INTEVAL = 100
C_UPDATE_STEP_INTERVAL = 1000
LR = 1E-4
EPSILON_MIN = 0.02
EPSILON_MAX = 1.0
EPSILON_PERIOD_DECAYING = 100000

# other operating parameters
n_save_model_episode_interval = 20


class DDQNAgent:
    def __init__(self, g, thread_idx, agent_type, actor, critic, data_num, action_size):
        self.thread_idx = thread_idx
        self.agent_type = agent_type
        self.data_num = data_num

        # self.global_actor = actor
        # self.global_critic = critic
        self.local_sess = tf.Session(graph=g)
        self.local_sess.run(tf.global_variables_initializer())

        self.actor = actor
        self.critic = critic

        # self.actor, self.critic = load_actor_critic_model(g=g, agent_type=self.agent_type)

        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]
        self.epsilon = 1.
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9995
        self.batch_size = 64
        self.action_size = action_size
        self.train_start = 1000
        self.target_update_interval = 10000

        # self.states = np.asarray()
        # self.actions = np.ndarray(shape=(0,2))
        # self.rewards = np.ndarray(shape=(0,1))
        # self.next_states = np.ndarray(shape=(0,))
        # self.dones = np.ndarray(shape=(0,1))

        # self.states = None
        # self.actions = None
        # self.rewards = None
        # self.next_states = None
        # self.dones = None

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self.discount_factor = 0.99

    # todo : this function can synchronize local network same as global network
    def update_target_model(self,  global_network, local_network):
        with self.local_sess as sess:
            global_network.set_weights(local_network.get_weights())

    def append_sample(self, state, action, reward, next_state, done):

        # self.actions = np.append(action, self.actions)
        # self.rewards = np.append(reward, self.rewards)
        # self.states = np.append(state, self.states)
        # self.next_states = np.append(next_state, self.next_states)
        # self.dones = np.append(done, self.dones)

        self.states.append(state)

        act = np.zeros(2)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def get_action(self, state):
        with self.local_sess as sess:
            policy = self.actor.predict(np.reshape(state, [1, self.state_size]))[0]  # todo : self.state_size
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def discount_rewards(self, done=True):
        discounted_rewards = np.zeros_like(self.rewards)
        running_add = 0
        if not done:
            pass
            # running_add = self.critic.predict(np.reshape(self.states[-1], (1, self.state_size)))[0]
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * 0.999 + self.rewards[t]  # todo : a3c discount_factor = 0.999
            discounted_rewards[t] = running_add
        return discounted_rewards

    # update policy network and value network every episode
    def train_episode(self, done):
        discounted_rewards = self.discount_rewards(done)

        inputs = []
        with self.local_sess as sess:
            for d_i in range(self.data_num):
                inputs.append([state[d_i] for state in self.states])
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
        with self.local_sess as sess:
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
        for i in range(self.data_num):
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
        for i in range(self.data_num):
            inputs.append(self.critic.input[i])
        inputs.append(discounted_reward)

        train = K.function(inputs, [], updates=updates)
        return train


class Agents(threading.Thread):
    agent_name = ['BSA', 'BOA', 'SSA', 'SOA']
    step_limit = [61, 60, 1, 0]
    additional_reward_rate = 0.1

    def __init__(self, idx, env, n_max_episode, file_dir):
        super(Agents, self).__init__()
        self.env = env
        self.n_max_episode = n_max_episode
        self.train_log_dir = file_dir
        self.idx = idx

        global SHARED_GRAPH

        bsa_actor, bsa_critic = load_actor_critic_model(g=SHARED_GRAPH, agent_type='bsa')
        boa_actor, boa_critic = load_actor_critic_model(g=SHARED_GRAPH, agent_type='boa')
        ssa_actor, ssa_critic = load_actor_critic_model(g=SHARED_GRAPH, agent_type='ssa')
        soa_actor, soa_critic = load_actor_critic_model(g=SHARED_GRAPH, agent_type='soa')

        bsa = DDQNAgent(SHARED_GRAPH, idx, 'bsa', bsa_actor, bsa_critic, data_num=2, action_size=2)
        boa = DDQNAgent(SHARED_GRAPH, idx, 'boa', boa_actor, boa_critic, data_num=3, action_size=2)
        ssa = DDQNAgent(SHARED_GRAPH, idx, 'ssa', ssa_actor, ssa_critic, data_num=4, action_size=2)
        soa = DDQNAgent(SHARED_GRAPH, idx, 'soa', soa_actor, soa_critic, data_num=4, action_size=2)

        self.agents = [bsa, boa, ssa, soa]
        self.sequence = 0

        self.boa_additional_data = [1, 1, 1, 1, 0, 0, 0]
        self.ssa_additional_data = [1, 1, 1, 1, 0, 0, 0]
        self.soa_additional_data = [1, 1, 1, 1, 0, 0, 0]

        self.sample_buffer = list()
        self.remain_step = 0
        self.trainable = False

        self.buy_price = 0
        self.sell_price = 0
        self.thread_step = 0

    def get_action(self, state):
        state = self._process_state(state)
        if self.sequence >= 1:
            self.remain_step -= 1
        if self.sequence >= 1 and self.remain_step <= self.step_limit[self.sequence]:  # limit 시간이 지나면 강제로 action = 1
            return 1
        return self.agents[self.sequence].get_action(state)

    # action 에 따라 bsa - boa - ssa - soa 순서를 진행한다.
    # 순서를 진행하면서 다음 agent에 필요한 additional_data 를 작성한다.
    # 순서가 넘어갈때 필요한 다른 모든것들을 여기서 처리한다.
    def _sequence_manage(self, action):
        self.trainable = False
        if self.sequence == 0 and action:
            # additional data 작성
            self.remain_step = 120
            self.sequence = 1

        elif self.sequence == 1 and action:
            # additional data 작성
            self.sequence = 2

        elif self.sequence == 2 and action:
            # additional data 작성
            self.sequence = 3

        elif self.sequence == 3 and action:
            # additional data 작성
            self.buffer_reward = self._append_buffer_sample()
            self.sample_buffer = list()
            self.sequence = 0
            self.trainable = True

    # agent 별로 state 가 다르기 때문에 (뒤로 갈수록 추가 정보가 생김) 그 처리를 한다.
    # additional data 자체는 sequence 가 넘어갈 때 _sequence_manage() 함수에서 생성하며, 이 함수는 그 additional data 를 state 에
    # 추가하는 역할을 한다.
    # 또한 agent 에 따라 input data 의 모양이 다르므로 그 처리도 여기서 한다.
    # agent 가 전부 구체화되면 완성할 것
    def _process_state(self, state):
        state = copy.deepcopy(list(state))
        # if self.sequence == 0:  # BSA
        #     state.append(self.time_to_binary_list(self.remain_step))  # < 테스트 후 지울것 (bsa 네트워크가 없어 boa 사용중이라 넣음)

        if self.sequence == 1 and not (self.boa_additional_data is None):  # BOA
            state.append(self.time_to_binary_list(self.remain_step))
            self.boa_time = self.remain_step

        if self.sequence == 2 and not (self.ssa_additional_data is None):  # SSA
            state.append(self.time_to_binary_list(self.boa_time - self.remain_step))
            state.append(self.time_to_binary_list(self.boa_time))
            self.ssa_time = self.remain_step

        if self.sequence == 3 and not (self.soa_additional_data is None):  # SOA
            state.append(self.time_to_binary_list(self.ssa_time))
            state.append(self.time_to_binary_list(self.ssa_time - self.remain_step))

        return np.array(state)

    @staticmethod
    def time_to_binary_list(time):
        bin_time = bin(time)[2:]
        for _ in range(7 - len(bin_time)):
            bin_time = '0' + bin_time
        bin_list = []
        for b in bin_time:
            bin_list.append(int(b))
        return np.asarray(bin_list)

    def append_sample(self, state, action, reward, next_state, done):
        state = self._process_state(state)
        next_state = self._process_state(next_state)

        if action == 0:  # action 이 0 인 경우 additional reward 가 없으므로 그냥 memory 에 sample 추가
            reward = 0  # action == 0 => reward = 0
            self.agents[self.sequence].append_sample(state, action, reward, next_state, done)
        else:  # action 이 1인 경우 additional reward 를 주기 위해 buffer 에 한번에 모았다가 reward 계산해서 마지막에 추가
            self.sample_buffer.append([state, action, reward[self.agent_name[self.sequence]], next_state, done])
            reward = reward[self.agent_name[self.sequence]]
        self._sequence_manage(action)
        # print('rb appended and size became {}'.format(len(self.sample_buffer)))
        return reward

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
            print('{} thread {} episode started'.format(self.idx, ep))
            episode_start_time = time.time()
            done = False
            state = self.env.reset()

            reward_sum = [0, 0, 0, 0]
            step_count = [0, 0, 0, 0]
            buy_count = 0
            profit = 0
            profit_comm = 0
            buy_price, sell_price = 0, 0
            commission = 0.33

            while not done:
                if self.sequence == 0:
                    self.env.remain_time = 120
                else:
                    if self.remain_step == 0:
                        self.env.remain_time = 1
                    else:
                        self.env.remain_time = self.remain_step
                action = self.get_action(state)
                if self.sequence == 0 and action == 1:
                    buy_count += 1
                    buy_price = self.env.holder_observation[-1][0] + 100
                if self.sequence == 3 and action == 1:
                    sell_price = self.env.holder_observation[-1][0] + 100
                    if buy_price != 0:
                        profit += (sell_price - buy_price) / buy_price
                        profit_comm += (sell_price - buy_price) / buy_price - commission * 0.01

                next_state, reward, done, info = self.env.step(action)
                prev_sequence = self.sequence

                if self.sequence == 3 and action == 1:
                    reward[self.agent_name[self.sequence]] = sell_price - buy_price - 0.33

                agent_reward = self.append_sample(state, action, reward, next_state, done)  #

                if action == 0:
                    reward_sum[prev_sequence] += agent_reward
                    step_count[prev_sequence] += 1

                if prev_sequence == 3 and action == 1:
                    for idx, br in enumerate(self.buffer_reward):
                        reward_sum[idx] += br
                        step_count[idx] += 1

                TOTAL_STEP_COUNT += 1
                state = next_state

                steps = 0
                for s in step_count:
                    steps += s

                # todo : need to change 1hr * 60 mins * 60 secs = 3600 secs - 120 secs needs to get enough observation for the first time
                if steps >= 1 * 60 * 60 - 3400:  # todo : -3000 for test
                    done = True
                    TOTAL_EPISODE = TOTAL_EPISODE + 1
                    for i, agent in enumerate(self.agents):
                        agent.train_episode(True)

            if step_count[0] > 0:
                rewards = []
                for i in range(4):
                    if step_count[i] == 0:
                        rewards.append(0)
                    else:
                        if i == 3:
                            pass
                        rewards.append(reward_sum[i]/step_count[i])

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
                print('bsa :', round(rewards[0], 5), end='  ')
                print('boa :', round(rewards[1], 5), end='  ')
                print('ssa :', round(rewards[2], 5), end='  ')
                print('soa :', round(rewards[3], 5))

                print('profit sum :', profit, end='  ')
                print('profit avg :', avg_profit, end='  ')
                print('profit(commission) sum :', profit_comm, end='  ')
                print('profit(commission) avg :', avg_profit_comm, end='  ')
                print('ep time :', int(time.time() - episode_start_time))

                # todo : it could occur resource deadlock, so that it could be reason being slow.. but for now, move on!
                train_log_file = open(self.train_log_dir, 'a', encoding='utf-8', newline='')
                train_log_writer = csv.writer(train_log_file)
                train_log_writer.writerow([ep, rewards[0], rewards[1], rewards[2], rewards[3], profit, avg_profit,
                                           profit_comm, avg_profit_comm])
                train_log_file.close()


class MyTGym(tgym.TradingGymEnv):  # MyTGym 수정해야 함 -> agent 별 reward 를 줘야 함 (4개 반환해서 agents 가 수정하거나 agent 입력해서 reward 주거나)
    # data shape
    rows = 10
    columns = 2
    seconds = 120
    channels = 2
    features = 11

    holder_observation = deque(np.array([[0 for x in range(52)] for y in range(seconds)]), maxlen=seconds)

    def _rewards(self, observation, action, done, info):
        rewards = {}
        secs = self.remain_time

        # create BSA reward
        width = 0
        price_at_signal = None
        threshold = 0.33
        for j in range(secs):
            if j == 0:
                price_at_signal = self.d_episodes_data[self.p_current_episode_ref_idx]['quote'].loc[
                    self.c_range_timestamp[self.p_current_step_in_episode]]['Price(last excuted)']  # 데이터 자체에 오타 나 있으므로 수정 x
            else:
                price = self.d_episodes_data[self.p_current_episode_ref_idx]['quote'].loc[self.c_range_timestamp[
                    self.p_current_step_in_episode+j]]['Price(last excuted)']
                gap = price - price_at_signal - threshold
                width += gap
        rewards['BSA'] = width / secs

        # create BOA rewrad

        low_price = price_at_signal
        for j in range(secs):
            price = self.d_episodes_data[self.p_current_episode_ref_idx]['quote'].loc[self.c_range_timestamp[
                self.p_current_step_in_episode+j]]['Price(last excuted)']
            if j == 0:
                current_price = price
            low_price = min(low_price, price)

        rewards['BOA'] = low_price - current_price

        # create SSA reward
        rewards['SSA'] = -width / secs

        # create SOA reward
        rewards['SOA'] = -width / secs
        return rewards

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
        env = MyTGym(episode_type='0', percent_goal_profit=2, percent_stop_loss=5, episode_duration_min=63)
        agent = Agents(i, env, N_MAX_EPISODE, train_log_file_dir)
        agents.append(agent)

    for a in agents:
        a.start()