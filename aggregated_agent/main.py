import os
import sys

newPath = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))+ '\\trading_gym'
sys.path.append(newPath)
sys.path.append('/home/lab4all/ag/trading_gym')
from gym_core import tgym
import numpy as np
import random
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Flatten, Concatenate, Input, LeakyReLU
from keras.optimizers import Adam
from collections import deque
import glob
import copy
from aggregated_agent import load
import csv
import time


class DDQNAgent:
    def __init__(self, agent_type, data_num, action_size):
        # load models
        self.agent_type = agent_type
        self.model = self.load_model()
        self.target_model = self.load_model()

        self.epsilon = 0.01
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 1024
        self.action_size = action_size
        self.train_start = 3000
        self.target_update_interval = 10000
        self.memory = deque(maxlen=7000)
        self.discount_factor = 0.99
        self.data_num = data_num

    def load_model(self):
        networks = glob.glob('./networks/*.h5f')
        if './networks/' + self.agent_type + '_rl.h5f' not in networks:
            trained_model = load.load_model(self.agent_type)

            for layer in trained_model.layers:
                layer.trainable = False
            rl_model = load.load_model(self.agent_type)
            concat_layer = Concatenate(name='concat2')([trained_model(rl_model.input), rl_model.layers[-1].output])
            output_layer = Dense(2, activation='linear', name='q_value_output')(concat_layer)
            model = Model(inputs=rl_model.input, outputs=output_layer)

        else:
            trained_model = load.load_model(self.agent_type)
            for layer in trained_model.layers:
                layer.trainable = False
            rl_model = load.load_model(self.agent_type)
            concat_layer = Concatenate(name='concat2')([trained_model(rl_model.input), rl_model.layers[-1].output])
            output_layer = Dense(2, activation='linear', name='q_value_output')(concat_layer)
            model = Model(inputs=rl_model.input, outputs=output_layer)
            model.load_weights('aggregated_agent/networks/' + self.agent_type + '_rl.h5f')
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model

    @staticmethod
    def pop_layer(model):
        if not model.outputs:
            raise Exception("model is empty")

        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            # print('ACTION')
            states = []
            for i in range(self.data_num):
                states.append(np.array([state[i]]))
            q_value = self.model.predict(states)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < self.train_start:
            # print('memory size is to short', len(self.memory))
            return
        # print('train', len(self.memory))
        # print('train', self.agent_type)

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

        # states = [np.array(states[0]), np.array(states[1]), np.array(states[2])]
        target = self.model.predict(input_states)
        target_val = self.target_model.predict(input_next_states)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        self.model.fit(input_states, target, batch_size=self.batch_size, epochs=1, verbose=0)
        self.model.save_weights('aggregated_agent/networks/' + self.agent_type + '_rl.h5f')


class Agents:
    agent_name = ['BSA', 'BOA', 'SSA', 'SOA']
    step_limit = [61, 60, 1, 0]
    additional_reward_rate = 0.1

    def __init__(self, bsa, boa, ssa, soa):
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
            self.boa_time = self.remain_step  # boa 액션시점 남은시간

        if self.sequence == 2 and not (self.ssa_additional_data is None):  # SSA
            state.append(self.time_to_binary_list(self.boa_time-self.remain_step))
            state.append(self.time_to_binary_list(self.boa_time))
            self.ssa_time = self.remain_step  # ssa 액션시점 남은시간

        if self.sequence == 3 and not (self.soa_additional_data is None):  # SOA
            state.append(self.time_to_binary_list(self.ssa_time))
            state.append(self.time_to_binary_list(self.ssa_time-self.remain_step))
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
        # print('state length :', len(state))
        # # print(state)
        state = self._process_state(state)
        next_state = self._process_state(next_state)
        # print('state length :', len(state))
        # print('next state length :', len(next_state))
        # print(state)
        # input()
        if self.sequence == 0:  # 지울것
            pass
            # print(round(reward[self.agent_name[self.sequence]], 3))
        if action == 0:  # action 이 0 인 경우 additional reward 가 없으므로 그냥 memory 에 sample 추가
            reward = 0  # action == 0 => reward = 0
            self.agents[self.sequence].append_sample(state, action, reward, next_state, done)

        else:  # action 이 1인 경우 additional reward 를 주기 위해 buffer 에 한번에 모았다가 reward 계산해서 마지막에 추가
            self.sample_buffer.append([state, action, reward[self.agent_name[self.sequence]], next_state, done])
            reward = reward[self.agent_name[self.sequence]]
        self._sequence_manage(action)
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
                    self.c_range_timestamp[self.p_current_step_in_episode]]['Price(last executed)']
            else:
                price = self.d_episodes_data[self.p_current_episode_ref_idx]['quote'].loc[self.c_range_timestamp[
                    self.p_current_step_in_episode+j]]['Price(last executed)']
                gap = price - price_at_signal - threshold
                width += gap
        rewards['BSA'] = width / secs

        # create BOA rewrad

        low_price = price_at_signal
        for j in range(secs):                
            price = self.d_episodes_data[self.p_current_episode_ref_idx]['quote'].loc[self.c_range_timestamp[
                    self.p_current_step_in_episode+j]]['Price(last executed)']
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
    env = MyTGym(episode_type='0', percent_goal_profit=2, percent_stop_loss=5, episode_duration_min=63)
    buy_signal_agent = DDQNAgent('bsa', data_num=2, action_size=2)
    buy_order_agent = DDQNAgent('boa', data_num=3, action_size=2)
    sell_signal_agent = DDQNAgent('ssa', data_num=4, action_size=2)
    sell_order_agent = DDQNAgent('soa', data_num=4, action_size=2)
    agents = Agents(buy_signal_agent, buy_order_agent, sell_signal_agent, sell_order_agent)
    # train_log_file = open('train_log.csv', 'a', encoding='utf-8', newline='')
    # csv_writer = csv.writer(train_log_file)

    EPISODES = 1000000
    for ep in range(EPISODES):
        try:
            start_time = time.time()

            done = False
            state = env.reset()
            agents.update_target_network()

            reward_sum = [0, 0, 0, 0]
            step_count = [0, 0, 0, 0]
            buy_count = 0
            profit = 0
            profit_comm = 0

            buy_price, sell_price = 0, 0
            commission = 0.33

            while not done:
                if agents.sequence == 0:
                    env.remain_time = 120
                else:
                    if agents.remain_step == 0:
                        env.remain_time = 1
                    else:
                        env.remain_time = agents.remain_step

                action = agents.get_action(state)
                if agents.sequence == 0 and action == 1:
                    buy_count += 1
                    buy_price = env.holder_observation[-1][0] + 100
                if agents.sequence == 3 and action == 1:
                    sell_price = env.holder_observation[-1][0] + 100
                    if buy_price != 0:
                        profit += (sell_price - buy_price) / buy_price
                        profit_comm += (sell_price - buy_price) / buy_price - commission * 0.01

                next_state, reward, done, info = env.step(action)
                prev_sequence = agents.sequence
                if agents.sequence == 3 and action == 1:
                    # print(sell_price, buy_price)
                    reward[agents.agent_name[agents.sequence]] = sell_price - buy_price - 0.33
                # print(reward)
                # print(reward)
                agent_reward = agents.append_sample(state, action, reward, next_state, done)  # 순서가 여기서 넘어감
                # if agents.sequence == 3:
                # print(agent_reward, end=' ')
                if action == 0:
                    reward_sum[prev_sequence] += agent_reward
                    step_count[prev_sequence] += 1

                if prev_sequence == 3 and action == 1:
                    for idx, br in enumerate(agents.buffer_reward):
                        reward_sum[idx] += br
                        step_count[idx] += 1

                state = next_state
                if agents.trainable and step_count[0] % 3 == 1:
                    agents.train_agents()
                steps = 0
                for s in step_count:
                    steps += s
                if steps >= 3600:
                    done = True

            if step_count[0] > 0:
                rewards = []
                for i in range(4):
                    if step_count[i] == 0:
                        rewards.append(0)
                    else:
                        if i == 3:
                            pass
                        # print(reward_sum[i])
                        # print(step_count[i])
                        rewards.append(reward_sum[i]/step_count[i])

                profit = round(profit * 100, 5)
                profit_comm = round(profit_comm * 100, 5)
                if buy_count == 0:
                    avg_profit, avg_profit_comm = 0, 0
                else:
                    avg_profit = round(profit / buy_count, 5)
                    avg_profit_comm = round(profit_comm / buy_count, 5)

                print('ep :', ep, end='  ')
                print('epsilon :', round(agents.agents[0].epsilon, 3), end='  ')
                print('buy :', buy_count)
                print('bsa :', round(rewards[0], 5), end='  ')
                print('boa :', round(rewards[1], 5), end='  ')
                print('ssa :', round(rewards[2], 5), end='  ')
                print('soa :', round(rewards[3], 5))

                print('profit sum :', profit, end='  ')
                print('profit avg :', avg_profit, end='  ')
                print('profit(commission) sum :', profit_comm, end='  ')
                print('profit avg :', avg_profit_comm, end='  ')
                print('ep time :', int(time.time() - start_time))

                train_log_file = open('./train_log.csv', 'a', encoding='utf-8', newline='')
                csv_writer = csv.writer(train_log_file)
                csv_writer.writerow([ep, rewards[0], rewards[1], rewards[2], rewards[3], profit, avg_profit, profit_comm, avg_profit_comm])
                train_log_file.close()
        except:
            pass



