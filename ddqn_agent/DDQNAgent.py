import core.tagent as tagent
import glob
import numpy as np
import random
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential, load_model

# action = [HOLD, BUY]

class DDQNAgent(tagent.TradingAgent):

    def __init__(self, state_size, action_size=2, file_dir='.', train_mode=True):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.file_dir = file_dir
        self.train_mode = train_mode
        self.model_name = '/dqn_type_01.h5'

        self.discount_factor = 0.999
        self.learning_rate = 0.0001
        self.epsilon = 1.
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.batch_size = 256
        self.train_start = 10000

        self.memory = deque(maxlen=100000)
        self.model = self._load_model()
        self.target_model = self._load_model()

        self.no_buy = 0

    def _load_model(self):
        item = glob.glob(self.file_dir + self.model_name)
        if len(item) == 0:
            return self._build_model()
        else:
            return load_model(item[0])

    def _build_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='relu', kernel_initializer='he_uniform'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def save_model(self):
        self.model.save(self.file_dir + self.model_name)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if self.no_buy > 0:
            self.no_buy -= 1
            return 0

        if self.train_mode:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            else:
                q_value = self.model.predict(state)
                agent_action = np.argmax(q_value[0])
                if agent_action == 1:
                    self.no_buy = 60
                return agent_action
        else:
            q_value = self.model.predict(state)
            agent_action = np.argmax(q_value[0])
            if agent_action == 1:
                self.no_buy = 60
            return agent_action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def train_model(self):
        self._decay_epsilon()

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

    def calc_reward(self, info):
        if info[0]['stop_loss']:
            return -1
        if info[0]['reached_profit']:
            return 1
        return 0


# EPISODES = 10000
# RENDER = False
# ACTION_SIZE = 2
#
# if __name__ == '__main__':
#     env = TradingGymEnv(episode_type=0)
#     agent = DQNAgent(state_size=env.get_state_size(), action_size=ACTION_SIZE)
#
#     for ep in range(EPISODES):
#         profit = 0
#         done = False
#         state = env.reset()
#
#         while not done:
#             if RENDER:
#                 env.render()
#
#             action = agent.get_action(state)
#             next_state, reward, done, info = env.step(action)
#
#             agent.append_sample(state, action, reward, next_state, done)
#             if len(agent.memory) >= agent.train_start:
#                 agent.train_model()
#
#             state = next_state
#             profit += reward
#
#             if done:
#                 agent.update_target_model()
#                 agent.save_model()
#                 print('profit :', profit, 'memory :', len(agent.memory), 'epsilon :', round(agent.epsilon, 5))