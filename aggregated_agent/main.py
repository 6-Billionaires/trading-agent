import os
import sys

newPath = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))+ '\\trading-gym'
sys.path.append(newPath)

from gym_core import tgym
import numpy as np
import random
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Flatten, Concatenate, Input
from keras.optimizers import Adam
from collections import deque
import glob
import copy


class DDQNAgent:
    def __init__(self, agent_type, state_size, action_size):
        # load models
        self.agent_type = agent_type
        self.model = self.load_model()
        self.target_model = self.load_model()

        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9999
        self.batch_size = 32
        self.state_size = state_size
        self.action_size = action_size
        self.train_start = 33
        self.target_update_interval = 10000
        self.memory = deque(maxlen=100000)
        self.discount_factor = 0.999

    def load_model(self):
        networks = glob.glob('./networks/*.h5')
        if './networks/' + self.agent_type + '_rl' not in networks:
            # model = load_model('./networks/' + self.agent_type + '.h5')
            # model.layers.pop()
            # output_layer = Dense(2, activation='linear', name='rl_output')(model.layers[-1].output)
            # model = Model(inputs=model.input, outputs=output_layer)

            trained_model = load_model('./networks/' + self.agent_type + '.h5')
            for layer in trained_model.layers:
                layer.trainable = False
            rl_model = load_model('./networks/' + self.agent_type + '.h5')
            concat_layer = Concatenate(name='concat2')([trained_model(rl_model.input), rl_model.layers[-1].output])
            output_layer = Dense(2, activation='linear', name='q_value_output')(concat_layer)
            model = Model(inputs=rl_model.input, outputs=output_layer)

        else:
            model = load_model('./networks/' + self.agent_type + '_rl.h5')

        # for layer in model.layers[:-1]:
        #     layer.trainable = False

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
            q_value = self.model.predict([np.array([state[0]]), np.array([state[1]]), np.array([state[2]])])
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < self.train_start:
            # print('memory size is to short', len(self.memory))
            return
        # print('train', len(self.memory))

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        mini_batch = random.sample(self.memory, self.batch_size)
        states = [[], [], []]
        next_states = [[], [], []]
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            for array_idx in range(3):
                states[array_idx].append(mini_batch[i][0][array_idx])
                next_states[array_idx].append(mini_batch[i][3][array_idx])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            dones.append(mini_batch[i][4])

        states = [np.array(states[0]), np.array(states[1]), np.array(states[2])]
        target = self.model.predict(states)
        target_val = self.target_model.predict([np.array(next_states[0]), np.array(next_states[1]), np.array(next_states[2])])

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)
        self.model.save('./networks/' + self.agent_type + '_rl.h5')


class Agents:
    agent_name = ['BSA', 'BOA', 'SSA', 'SOA']
    step_limit = [60, 59, 1, 0]
    additional_reward_rate = 0.1

    def __init__(self, bsa, boa, ssa, soa):
        self.agents = [bsa, boa, ssa, soa]
        self.sequence = 0

        self.boa_additional_data = [1, 1, 1, 1, 1, 1, 1]
        self.ssa_additional_data = [1, 1, 1, 1, 1, 1, 1]
        self.soa_additional_data = [1, 1, 1, 1, 1, 1, 1]  # . 남은 시간의 이진 표현이 들어가는 자리 (테스트용임)

        self.sample_buffer = list()
        self.remain_step = 0

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
            self._append_buffer_sample()
            self.sample_buffer = list()
            self.sequence = 0

    # agent 별로 state 가 다르기 때문에 (뒤로 갈수록 추가 정보가 생김) 그 처리를 한다.
    # additional data 자체는 sequence 가 넘어갈 때 _sequence_manage() 함수에서 생성하며, 이 함수는 그 additional data 를 state 에
    # 추가하는 역할을 한다.
    # 또한 agent 에 따라 input data 의 모양이 다르므로 그 처리도 여기서 한다.
    # agent 가 전부 구체화되면 완성할 것
    def _process_state(self, state):
        state = copy.deepcopy(state)
        if self.sequence == 0:  # BSA
            state.append([1, 1, 1, 1, 1, 1, 1])
            # state = np.append(state, [1, 1, 1, 1, 1, 1, 1], axis=0)  # < 테스트 후 지울것 (bsa 네트워크가 없어 boa 사용중이라 넣음)
            pass

        if self.sequence == 1 and not (self.boa_additional_data is None):  # BOA
            state.append([1, 1, 1, 1, 1, 1, 1])
            # state = np.append(state, self.boa_additional_data, axis=0)

        if self.sequence == 2 and not (self.ssa_additional_data is None):  # SSA
            state.append([1, 1, 1, 1, 1, 1, 1])
            # state = np.append(state, self.ssa_additional_data, axis=0)

        if self.sequence == 3 and not (self.soa_additional_data is None):  # SOA
            state.append([1, 1, 1, 1, 1, 1, 1])
            # state = np.append(state, self.soa_additional_data, axis=0)

        return np.array(state)

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
            print(reward[self.agent_name[self.sequence]])
        if action == 0:  # action 이 0 인 경우 additional reward 가 없으므로 그냥 memory 에 sample 추가
            reward = 0  # action == 0 => reward = 0
            self.agents[self.sequence].append_sample(state, action, reward, next_state, done)

        else:  # action 이 1인 경우 additional reward 를 주기 위해 buffer 에 한번에 모았다가 reward 계산해서 마지막에 추가
            self.sample_buffer.append([state, action, reward[self.agent_name[self.sequence]], next_state, done])
            reward = reward[self.agent_name[self.sequence]]
        self._sequence_manage(action)
        return reward

    def _append_buffer_sample(self):
        self.sample_buffer[0][2] += (self.sample_buffer[1][2] + self.sample_buffer[3][2]) * self.additional_reward_rate
        self.sample_buffer[1][2] += self.sample_buffer[3][2] * self.additional_reward_rate
        self.sample_buffer[2][2] += self.sample_buffer[3][2] * self.additional_reward_rate
        sb = self.sample_buffer

        for idx, agent in enumerate(self.agents):
            agent.append_sample(sb[idx][0], sb[idx][1], sb[idx][2], sb[idx][3], sb[idx][4])

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
    seconds = 90
    channels = 2
    features = 11

    holder_observation = deque(np.array([[0 for x in range(52)] for y in range(seconds)]), maxlen=seconds)

    def _rewards(self, observation, action, done, info):
        rewards = {}
        secs = 60

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
        rewards['BOA'] = width / secs

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
    env = MyTGym(episode_type='0', percent_goal_profit=2, percent_stop_loss=5, episode_duration_min=60)
    buy_signal_agent = DDQNAgent('bsa', 3120, 2)  # buy signal agent 의 .h5 파일 경로, state 수, action 수
    buy_order_agent = DDQNAgent('boa', 3120, 2)  # buy order agent 의 .h5 파일 경로, state 수, action 수
    sell_signal_agent = DDQNAgent('ssa', 3120, 2)  # sell signal agent 의 .h5 파일 경로, state 수, action 수
    sell_order_agent = DDQNAgent('soa', 3120, 2)  # sell order agent 의 .h5 파일 경로, state 수, action 수
    agents = Agents(buy_signal_agent, buy_order_agent, sell_signal_agent, sell_order_agent)

    EPISODES = 1000000
    for ep in range(EPISODES):
        done = False
        state = env.reset()
        agents.update_target_network()

        reward_sum = 0
        step_count = 0

        while not done:
            action = agents.get_action(state)
            next_state, reward, done, info = env.step(action)
            reward_sum += agents.append_sample(state, action, reward, next_state, done)
            step_count += 1
            state = next_state
            agents.train_agents()

        print('step :', step_count)
        if step_count > 0:
            print('reward :', reward_sum / step_count)







