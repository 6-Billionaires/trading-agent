import argparse
import config

parser = argparse.ArgumentParser()
parser.add_argument("-import-gym", "--import-gym", help="import trading gym", action="store_true")
parser.add_argument("-gym-dir", "--gym-dir", type=str, help="import trading gym")
args = parser.parse_args()

if args.import_gym:
    import sys
    sys.path.insert(0, args.gym_dir)

from keras.models import Model
from keras.layers import LeakyReLU, Input, Dense, Conv3D, Conv1D, Dense, Flatten, MaxPooling1D, MaxPooling2D, MaxPooling3D,Concatenate
import numpy as np
import pickle
from gym_core.ioutil import *  # file i/o to load stock csv files
from core import util
import tensorflow as tf
import keras.backend as K

_len_observation = 120


def load_model(g):
    agent_type = 'single_a3c_agent'
    with g.as_default():
        networks = glob.glob('networks/*.h5f')
        actor_file = 'networks/' + agent_type + '_actor.h5f'
        critic_file = 'networks/' + agent_type + '_critic.h5f'

        actor, critic = build_network(activation='leaky_relu', neurons=100)
        if 'networks/' + agent_type + '_actor.h5f' not in networks \
                or 'networks/' + agent_type + '_critic.h5f' not in networks:
            actor.load_weights(actor_file)
            critic.load_weights(critic_file)

        actor.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'])
        critic.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'])
        return actor, critic


def build_network(optimizer='adam',init_mode='uniform', filters=16, neurons=20, activation='relu'):
    if activation == 'leaky_relu':
        activation = LeakyReLU(alpha=0.3)

    input_order = Input(shape=(10, 2, _len_observation, 2), )
    input_tranx = Input(shape=(_len_observation, 11), )

    h_conv1d_2 = Conv1D(filters=16, kernel_initializer=init_mode, kernel_size=3)(input_tranx)
    h_conv1d_2 = LeakyReLU(alpha=0.3)(h_conv1d_2)
    h_conv1d_4 = MaxPooling1D(pool_size=3,  strides=None, padding='valid')(h_conv1d_2)
    h_conv1d_6 = Conv1D(filters=32, kernel_initializer=init_mode, kernel_size=3)(h_conv1d_4)
    h_conv1d_6 = LeakyReLU(alpha=0.3)(h_conv1d_6)
    h_conv1d_8 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(h_conv1d_6)

    h_conv3d_1_1 = Conv3D(filters=filters, kernel_initializer=init_mode, kernel_size=(2, 1, 5))(input_order)
    h_conv3d_1_1 = LeakyReLU(alpha=0.3)(h_conv3d_1_1)
    h_conv3d_1_2 = Conv3D(filters=filters,  kernel_initializer=init_mode,kernel_size=(1, 2, 5))(input_order)
    h_conv3d_1_2 = LeakyReLU(alpha=0.3)(h_conv3d_1_2)

    h_conv3d_1_3 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_1)
    h_conv3d_1_4 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_2)

    h_conv3d_1_5 = Conv3D(kernel_initializer=init_mode, filters=filters*2, kernel_size=(1, 2, 5))(h_conv3d_1_3)
    h_conv3d_1_5 = LeakyReLU(alpha=0.3)(h_conv3d_1_5)

    h_conv3d_1_6 = Conv3D(kernel_initializer=init_mode, filters=filters*2, kernel_size=(2, 1, 5))(h_conv3d_1_4)
    h_conv3d_1_6 = LeakyReLU(alpha=0.3)(h_conv3d_1_6)

    h_conv3d_1_7 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_5)
    h_conv3d_1_8 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_6)
    o_conv3d_1 = Concatenate(axis=-1)([h_conv3d_1_7, h_conv3d_1_8])

    o_conv3d_1_1 = Flatten()(o_conv3d_1)

    i_concatenated_all_h_1 = Flatten()(h_conv1d_8)

    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1])

    shared = Dense(neurons, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

    actor_output = Dense(1, kernel_initializer=init_mode, activation='linear')(shared)
    critic_output = Dense(3, kernel_initializer=init_mode, activation='softmax')(shared)

    actor = Model([input_order, input_tranx], actor_output)
    critic = Model([input_order, input_tranx], critic_output)

    actor.compile(optimizer='adam', loss='mse')
    critic.compile(optimizer='adam', loss='mse')

    actor.summary()
    critic.summary()

    return actor, critic
