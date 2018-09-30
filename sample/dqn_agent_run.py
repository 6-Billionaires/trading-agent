from gym_core import tgym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy


EPISODES = 10000
RENDER = False
ACTION_SIZE = 2
OBSERVATION_SIZE = 111

class myTGym(tgym.TradingGymEnv):
    def _rewards(self, observation, action, done, info):
        if action == 1:
            if info['stop_loss']:
                return -1
            if info['reached_profit']:
                return 1
        return 0

def build_network():

    model = Sequential()
    model.add( Flatten(input_shape=(1,)  + (111,) )  )
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(30))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('linear'))
    # print(model.summary())
    print(model.output._keras_shape)

    return model


if __name__ == '__main__':
    env = myTGym(episode_type='0', percent_goal_profit=2, percent_stop_loss=5)
    # s1, s2, s3 = env.reset()
    # state = aggregate_state(s1, s2, s3)

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    model  = build_network()

    dqn = DQNAgent(model=model, nb_actions=2, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)

    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format('trading'), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)
