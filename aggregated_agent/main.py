import os
import sys

newPath = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))+ '\\trading-gym'
sys.path.append(newPath)

from gym_core import tgym

import numpy as np
import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from collections import deque


class DDQNAgent:
    def __init__(self, model_dir, state_size, action_size):
        # load models
        self.model = load_model(model_dir)
        self.target_model = load_model(model_dir)
        self.model_dir = model_dir

        self.epsilon = 1.
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9999
        self.batch_size = 32
        self.state_size = state_size
        self.action_size = action_size
        self.train_start = 33
        self.target_update_interval = 10000
        self.memory = deque(maxlen=100000)
        self.discount_factor = 0.999

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < self.train_start:
            print('memory size is to short', len(self.memory))
            return
        print('train', len(self.memory))

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = list(), list(), list()

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)
        self.model.save(self.model_dir)


class Agents:
    agent_name = ['BSA', 'BOA', 'SSA', 'SOA']
    step_limit = [60, 59, 1, 0]
    additional_reward_rate = 0.1

    def __init__(self, bsa, boa, ssa, soa):
        self.agents = [bsa, boa, ssa, soa]
        self.sequence = 0

        self.boa_additional_data = None
        self.ssa_additional_data = None
        self.soa_additional_data = None

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
        if self.sequence == 0:  # BSA
            pass

        if self.sequence == 1 and not (self.boa_additional_data is None):  # BOA
            state = np.append(state, self.boa_additional_data)

        if self.sequence == 2 and not (self.ssa_additional_data is None):  # SSA
            state = np.append(state, self.ssa_additional_data)

        if self.sequence == 3 and not (self.soa_additional_data is None):  # SOA
            state = np.append(state, self.soa_additional_data)

        return state

    def append_sample(self, state, action, reward, next_state, done):
        if self.sequence == 0:
            print(reward[self.agent_name[self.sequence]])
        if action == 0:  # action 이 0 인 경우 additional reward 가 없으므로 그냥 memory 에 sample 추가
            self.agents[self.sequence].append_sample(state, action, reward[self.agent_name[self.sequence]], next_state,
                                                     done)
        else:  # action 이 1인 경우 additional reward 를 주기 위해 buffer 에 한번에 모았다가 reward 계산해서 마지막에 추가
            self.sample_buffer.append([state, action, reward[self.agent_name[self.sequence]], next_state, done])
        self._sequence_manage(action)

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
    holder_observation = deque(np.array([[0 for x in range(52)] for y in range(60)]), maxlen=60)

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
                    self.c_range_timestamp[self.p_current_step_in_episode]]['Price(last excuted)']
            else:
                price = self.d_episodes_data[self.p_current_episode_ref_idx]['quote'].loc[self.c_range_timestamp[
                    self.p_current_step_in_episode+j]]['Price(last excuted)']
                gap = price - price_at_signal - threshold
                width += gap
        rewards['BSA'] = width

        # create BOA rewrad
        rewards['BOA'] = [0.1, 0, -0.4, 0.3][random.randint(0, 3)]

        # create SSA reward
        rewards['SSA'] = [0.1, 0, -0.4, 0.3][random.randint(0, 3)]

        # create SOA reward
        rewards['SOA'] = [0.1, 0, -0.4, 0.3][random.randint(0, 3)]
        return rewards

    def observation_processor(self, observation):
        observation = {}

        # create BSA observation
        return np.array([0 for x in range(3120)])  # 임시 observation

if __name__ == '__main__':
    env = MyTGym(episode_type='0', percent_goal_profit=2, percent_stop_loss=5, episode_duration_min=60)
    buy_signal_agent = DDQNAgent('./agents/test1.h5', 3120, 2)  # buy signal agent 의 .h5 파일 경로, state 수, action 수
    buy_order_agent = DDQNAgent('./agents/test2.h5', 3120, 2)  # buy order agent 의 .h5 파일 경로, state 수, action 수
    sell_signal_agent = DDQNAgent('./agents/test3.h5', 3120, 2)  # sell signal agent 의 .h5 파일 경로, state 수, action 수
    sell_order_agent = DDQNAgent('./agents/test4.h5', 3120, 2)  # sell order agent 의 .h5 파일 경로, state 수, action 수
    agents = Agents(buy_signal_agent, buy_order_agent, sell_signal_agent, sell_order_agent)

    EPISODES = 1000000
    for ep in range(EPISODES):
        done = False
        state = env.reset()
        agents.update_target_network()

        while not done:
            action = agents.get_action(state)
            next_state, reward, done, info = env.step(action)
            agents.append_sample(state, action, reward, next_state, done)
            state = next_state
            agents.train_agents()






