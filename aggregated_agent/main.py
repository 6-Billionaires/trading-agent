"""
faster dqn aggregated agent for stock skelping trading!!
    . it is to make original dqn agent faster and make sure that we could do experiment in fast fail mode

0. algorithms
    . learner is only one and actor could have as many as n_threads configured
    . learner will update target network if total_step_count, global shared variable % update_step_interval  == 0

1. objects
    . agent(env, bsa, boa, ssa, soa)
    . faster-dqn-agent

2. global shared variables between threads
    . total_step_count

    . total_episode

    . scores

    . nw_bsa
        at first, just agent also use single same policy and target network
    . nw_boa
        at first, just agent also use single same  policy and target network
    . nw_ssa
        at first, just agent also use single same policy and target network
    . nw_soa
        at first, just agent also use single same policy and target network
    . t_nw_bsa
    . t_nw_boa
    . t_nw_ssa
    . t_nw_soa

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
from aggregated_agent import load
import csv
import threading
import  tensorflow as tf
from keras import backend as K

class DDQNAgent:
    def __init__(self, agent_type, model, target_model, data_num, action_size, rb):
        # load models
        self.agent_type = agent_type
        self.model = model
        self.target_model = target_model
        # self.model = self.load_model()
        # self.target_model = self.load_model()

        self.epsilon = 0.3
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9999
        self.batch_size = 64
        self.action_size = action_size
        self.train_start = 1000
        self.target_update_interval = 10000
        self.memory = rb
        self.discount_factor = 0.999
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
            states = []
            for i in range(self.data_num):
                states.append(np.array([state[i]]))
            with graph.as_default():
                q_value = self.model.predict(states)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):

        # todo : here self.train_start is same as c_warm_up_step hyperparameter!!!
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

        with graph.as_default():
            target = self.model.predict(input_states)
            target_val = self.target_model.predict(input_next_states)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        lock_update_policy_nw.acquire()
        with graph.as_default():
            self.model.fit(input_states, target, batch_size=self.batch_size, epochs=1, verbose=0)
        lock_update_policy_nw.release()

        if total_episode % n_save_model_episode_interval == 0:
            with graph.as_default():
                self.model.save_weights('aggregated_agent/networks/' + self.agent_type + '_rl.h5f')


def load_model(agent_type):
    networks = glob.glob('./networks/*.h5f')
    if './networks/' + agent_type + '_rl.h5f' not in networks:
        trained_model = load.load_model(agent_type)
        for layer in trained_model.layers:
            layer.trainable = False
        rl_model = load.load_model(agent_type)
        # rl_model = load_model('./networks/' + self.agent_type + '.h5')
        concat_layer = Concatenate(name='concat2')([trained_model(rl_model.input), rl_model.layers[-1].output])
        output_layer = Dense(2, activation='linear', name='q_value_output')(concat_layer)
        model = Model(inputs=rl_model.input, outputs=output_layer)

    else:
        trained_model = load.load_model(agent_type)
        for layer in trained_model.layers:
            layer.trainable = False
        rl_model = load.load_model(agent_type)
        concat_layer = Concatenate(name='concat2')([trained_model(rl_model.input), rl_model.layers[-1].output])
        output_layer = Dense(2, activation='linear', name='q_value_output')(concat_layer)
        model = Model(inputs=rl_model.input, outputs=output_layer)
        model.load_weights('aggregated_agent/networks/' + agent_type + '_rl.h5f')
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model



# shared global variables
total_step_count = 0
total_episode = 0
scores = []
target_network = None

# hyperparameter
c_rb_size = 100000
c_warm_up_step = 10000
c_update_step_interval = 1000
lr = 1e-4
epsilon_min = 0.02
epsilon_max = 1.0
epsilon_period_decaying = 100000

# other operating parameters
n_save_model_episode_interval = 20
tf.reset_default_graph()
graph = tf.get_default_graph()  # todo : graph is not thread safe

# self.sess = tf.InteractiveSession()
# K.set_session(self.sess)
# self.sess.run(tf.global_variables_initializer())


# this is a replay memory of buy signal agent
rb_bsa = deque(maxlen=c_rb_size)
# this is a replay memory of buy order agent
rb_boa = deque(maxlen=c_rb_size)
# this is a replay memory of sell signal agent
rb_ssa = deque(maxlen=c_rb_size)
# this is a replay memory of sell order agent
rb_soa = deque(maxlen=c_rb_size)

# check lock when thread agent try appending state transition information into replay memory.
lock_rb_append = threading.Lock()
# check lock when thread agent try updating policy network using target network
lock_update_policy_nw = threading.Lock()


nw_bsa = load_model(agent_type='bsa')
t_nw_bsa = load_model(agent_type='bsa')

nw_boa = load_model(agent_type='boa')
t_nw_boa = load_model(agent_type='boa')

nw_ssa = load_model(agent_type='ssa')
t_nw_ssa = load_model(agent_type='ssa')

nw_soa = load_model(agent_type='soa')
t_nw_soa = load_model(agent_type='soa')

bsa = DDQNAgent('bsa', model=nw_bsa, target_model=t_nw_bsa, data_num=2, action_size=2, rb=rb_bsa)
boa = DDQNAgent('boa', model=nw_boa, target_model=t_nw_boa, data_num=3, action_size=2, rb=rb_boa)
ssa = DDQNAgent('ssa', model=nw_ssa, target_model=t_nw_ssa, data_num=4, action_size=2, rb=rb_ssa)
soa = DDQNAgent('soa', model=nw_soa, target_model=t_nw_soa, data_num=4, action_size=2, rb=rb_soa)


class Agents(threading.Thread):
    agent_name = ['BSA', 'BOA', 'SSA', 'SOA']
    step_limit = [61, 60, 1, 0]
    additional_reward_rate = 0.1

    def __init__(self, env, n_max_episode, csv_writer, _bsa, _boa, _ssa, _soa):
        super(Agents, self).__init__()
        self.env = env
        self.n_max_episode = n_max_episode
        self.cw = csv_writer

        global bsa
        global boa
        global ssa
        global soa

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
            self._append_buffer_sample()
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

        if self.sequence == 2 and not (self.ssa_additional_data is None):  # SSA
            state.append(self.time_to_binary_list(self.remain_step))
            state.append(self.time_to_binary_list(120-self.remain_step))

        if self.sequence == 3 and not (self.soa_additional_data is None):  # SOA
            state.append(self.time_to_binary_list(self.remain_step))
            state.append(self.time_to_binary_list(120-self.remain_step))

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

    # Thread interactive with environment
    def run(self):
        global total_episode
        global graph
        for ep in range(self.n_max_episode):
            done = False
            state = self.env.reset()

            if self.thread_step % c_update_step_interval == 0:
                with graph.as_default():
                    self.update_target_network()

            reward_sum = 0
            step_count = 0
            buy_count = 0
            profit = 0
            profit_comm = 0
            buy_price, sell_price = 0, 0
            commission = 0.33

            while not done:
                action = self.get_action(state)
                if self.sequence == 0 and action == 1:
                    buy_count += 1
                    buy_price = self.env.holder_observation[-1][0]
                if self.sequence == 3 and action == 1:
                    sell_price = self.env.holder_observation[-1][0]
                    if buy_price != 0:
                        profit += (sell_price - buy_price) / buy_price
                        profit_comm += (sell_price - buy_price) / buy_price - commission * 0.01

                next_state, reward, done, info = self.env.step(action)
                lock_rb_append.acquire()
                agent_reward = self.append_sample(state, action, reward, next_state, done)
                lock_rb_append.release()

                reward_sum += agent_reward

                step_count += 1
                state = next_state
                if self.trainable:
                    self.train_agents()
                # todo : here we need to change 1 hour * 60 minutes * 60 seconds = 3600 seconds
                if step_count >= 1 * 60 * 60:
                    done = True
                    total_episode = total_episode + 1

            if step_count > 0:
                avg_reward = round(reward_sum / step_count, 7)
                profit = round(profit * 100, 5)
                profit_comm = round(profit_comm * 100, 5)
                if buy_count == 0:
                    avg_profit, avg_comm_profit = 0, 0
                else:
                    avg_profit = profit / buy_count
                    avg_comm_profit = profit_comm / buy_count
                print('ep :', ep, end='  ')
                print('epsilon :', round(self.agents[0].epsilon, 3), end='  ')
                print('avg reward :', avg_reward, end='  ')
                print('buy :', buy_count, end='  ')
                print('profit :', round(profit, 3), end='  ')
                print('avg profit :', round(avg_profit, 3), end='  ')
                print('profit(comm) :', round(profit_comm, 3), end='  ')
                print('avg profit(comm) :', round(avg_comm_profit, 3))

                # todo : it could occur resource deadlock, so that it could be reason being slow.. but for now, move on!
                self.cw.writerow([ep, avg_reward, buy_count, profit, avg_profit, profit_comm, avg_comm_profit])


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


class FasterDQNAgent:
    def __init__(self, n_threads, n_max_episode):
        # todo : add hyperparameters here
        self.n_threads = n_threads
        self.n_max_episode = n_max_episode

    def play(self):
        faster_dqn_agents = []
        train_log_file = open('train_log.csv', 'a', encoding='utf-8', newline='')
        csv_writer = csv.writer(train_log_file)

        for i in range(self.n_threads):
            env = MyTGym(episode_type='0', percent_goal_profit=2, percent_stop_loss=5, episode_duration_min=60)
            agent = Agents(env, 10000, csv_writer, bsa, boa, ssa, soa)
            faster_dqn_agents.append(agent)

        for a in faster_dqn_agents:
            a.start()


if __name__ == '__main__':
    fa = FasterDQNAgent(2, 40)  # thread, n_max_episdoe
    fa.play()