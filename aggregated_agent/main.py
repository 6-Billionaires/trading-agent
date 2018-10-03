"""
faster dqn aggregated agent for stock scalping trading!!
    . it is to make original dqn agent faster and make sure that we could do experiment in fast fail mode
    . we can give threads count as much as server allows
    . there is only one policy network update by all threads. one only thread update at once thorugh lock
    . threads could gather samples

0. algorithms
    . learner is only one and actor could have as many as n_threads configured
    . learner will update target network if total_step_count, global shared variable % update_step_interval  == 0

1. objects
    . agent(env, bsa, boa, ssa, soa)
    . faster-dqn-agent

2. global shared variables between threads
    . total_step_count
        across threads, it represents total step counts
    . total_episode
        across threads, it represents total episodes
    . scores
        across threads, it represents all scores from all episodes over all threads running.
    . nw_bsa
        at first, just agent also use single same policy and target network
    . nw_boa
        at first, just agent also use single same  policy and target network
    . nw_ssa
        at first, just agent also use single same policy and target network
    . nw_soa
        at first, just agent also use single same policy and target network
    . t_nw_bsa
        target network version buy signal agent
    . t_nw_boa
        target network version buy order agent
    . t_nw_ssa
        target network version sell signal agent
    . t_nw_soa
        target network version sell order agent

    . rb_bsa
        this is a replay memory of buy signal agent
    . rb_boa
        this is a replay memory of buy order agent
    . rb_ssa
        this is a replay memory of sell signal agent
    . rb_soa
        this is a replay memory of sell order agent

3. threading lock
    . check lock when thread agent try appending state transition information into replay memory.
    . check lock when thread agent try updating policy network using target network so that,
        if target network is updating by learner, it will wait until update is finished.


4. Hyperparameter setting
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
from aggregated_agent.loadnetwork import *
from aggregated_agent import loadnetwork
import csv
import threading
import  tensorflow as tf
from keras import backend as K
import time

SHARED_GRAPH = tf.get_default_graph()  # todo : graph is not thread safe
N_THREADS = 16
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

# this is a replay memory of buy signal agent
RB_BSA = deque(maxlen=C_RB_SIZE)
# this is a replay memory of buy order agent
RB_BOA = deque(maxlen=C_RB_SIZE)
# this is a replay memory of sell signal agent
RB_SSA = deque(maxlen=C_RB_SIZE)
# this is a replay memory of sell order agent
RB_SOA = deque(maxlen=C_RB_SIZE)

NW_BSA = load_rl_model(g=SHARED_GRAPH, agent_type='bsa', i=1)
T_NW_BSA = load_rl_model(g=SHARED_GRAPH, agent_type='bsa', i=11)
NW_BOA = load_rl_model(g=SHARED_GRAPH, agent_type='boa', i=2)
T_NW_BOA = load_rl_model(g=SHARED_GRAPH, agent_type='boa', i=22)
NW_SSA = load_rl_model(g=SHARED_GRAPH, agent_type='ssa', i=3)
T_NW_SSA = load_rl_model(g=SHARED_GRAPH, agent_type='ssa', i=33)
NW_SOA = load_rl_model(g=SHARED_GRAPH, agent_type='soa', i=4)
T_NW_SOA = load_rl_model(g=SHARED_GRAPH, agent_type='soa', i=44)

class DDQNAgent:
    def __init__(self, thread_idx, agent_type, policy_model, target_model,
                 data_num, action_size, rb, use_thread_model=False):
        self.thread_idx = thread_idx
        self.agent_type = agent_type

        if use_thread_model:
            # self.model = self.load_model()
            self.model = policy_model
            self.target_model = self.load_model()
        else:
            self.model = policy_model
            self.target_model = target_model

        self.epsilon = 1.
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9995
        self.batch_size = 64
        self.action_size = action_size
        self.train_start = 1000
        self.target_update_interval = 10000
        self.memory = rb
        self.discount_factor = 0.99
        self.data_num = data_num

    def update_target_model(self):
        with SHARED_GRAPH.as_default():
            self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            states = []
            for i in range(self.data_num):
                states.append(np.array([state[i]]))
            with SHARED_GRAPH.as_default():
                q_value = self.model.predict(states)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        # todo : here self.train_start is same as c_warm_up_step hyper-parameter!!!
        if len(self.memory) < self.train_start:
            return
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        mini_batch = random.sample(self.memory, self.batch_size)

        states, next_states = [], []
        for _ in range(self.data_num):
            states.append([])
            next_states.append([])
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            for array_idx in range(self.data_num):
                states[array_idx].append(mini_batch[i][0][array_idx])
                next_states[array_idx].append(mini_batch[i][3][array_idx])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            dones.append(mini_batch[i][4])

        input_states, input_next_states = [], []
        for i in range(self.data_num):
            input_states.append(np.array(states[i]))
            input_next_states.append(np.array(next_states[i]))

        with SHARED_GRAPH.as_default():
            target = self.model.predict(input_states)
            target_val = self.target_model.predict(input_next_states)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        with SHARED_GRAPH.as_default():
            self.model.fit(input_states, target, batch_size=self.batch_size, epochs=1, verbose=0)

        if TOTAL_EPISODE % n_save_model_episode_interval == 0:
            pass
            # with graph.as_default():
            #     self.model.save_weights('aggregated_agent/networks/' + self.agent_type + '_rl.h5f')

class Agents(threading.Thread):
    agent_name = ['BSA', 'BOA', 'SSA', 'SOA']
    step_limit = [61, 60, 1, 0]
    additional_reward_rate = 0.1

    def __init__(self, idx, n_max_episode, file_dir, nw_bsa, nw_boa, nw_ssa, nw_soa,
                 t_nw_bsa, t_nw_boa, t_nw_ssa, t_nw_soa, rb_bsa, rb_boa, rb_ssa, rb_soa):
        super(Agents, self).__init__()
        env = MyTGym(episode_type='0', percent_goal_profit=2, percent_stop_loss=5, episode_duration_min=63)
        self.env = env
        self.n_max_episode = n_max_episode
        self.train_log_dir = file_dir
        self.idx = idx

        global TOTAL_STEP_COUNT

        bsa = DDQNAgent(idx, 'bsa', policy_model=nw_bsa, target_model=t_nw_bsa, data_num=2, action_size=2, rb=rb_bsa)
        boa = DDQNAgent(idx, 'boa', policy_model=nw_boa, target_model=t_nw_boa, data_num=3, action_size=2, rb=rb_boa)
        ssa = DDQNAgent(idx, 'ssa', policy_model=nw_ssa, target_model=t_nw_ssa, data_num=4, action_size=2, rb=rb_ssa)
        soa = DDQNAgent(idx, 'soa', policy_model=nw_soa, target_model=t_nw_soa, data_num=4, action_size=2, rb=rb_soa)

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
        return bin_list

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

    def update_target_network(self):
        for agent in self.agents:
            agent.update_target_model()

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

            if self.idx == 0 and TOTAL_STEP_COUNT % C_SYNC_STEP_INTEVAL == 0:
                self.update_target_network()

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

                agent_reward = self.append_sample(state, action, reward, next_state, done) #

                if action == 0:
                    reward_sum[prev_sequence] += agent_reward
                    step_count[prev_sequence] += 1

                if prev_sequence == 3 and action == 1:
                    for idx, br in enumerate(self.buffer_reward):
                        reward_sum[idx] += br
                        step_count[idx] += 1

                TOTAL_STEP_COUNT += 1
                state = next_state
                # todo : add thread 0 only update policy network!!
                if self.idx == 0 and self.trainable and steps % C_UPDATE_STEP_INTERVAL == 0:
                    self.train_agents()

                steps = 0
                for s in step_count:
                    steps += s
                # todo : need to change 1hr * 60 mins * 60 secs = 3600 secs - 120 secs needs to get enough observation for the first time
                if steps >= 1 * 60 * 60 - 3300:  # todo : -3000 for test
                    done = True
                    TOTAL_EPISODE = TOTAL_EPISODE + 1

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
    faster_dqn_agents = []

    for i in range(N_THREADS):
        agent = Agents(i, N_MAX_EPISODE, train_log_file_dir, NW_BSA, NW_BOA, NW_SSA, NW_SOA, T_NW_BSA,
                       T_NW_BOA, T_NW_SSA, T_NW_SOA, RB_BSA, RB_BOA, RB_SSA, RB_SOA)
        faster_dqn_agents.append(agent)

    sess = K.get_session()
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    for a in faster_dqn_agents:
        a.start()