
import argparse
import config

parser = argparse.ArgumentParser()
parser.add_argument("-import-gym", "--import-gym",help="import trading gym", action="store_true")
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


def load_model(model_type):
    if model_type == 'bsa':
        model = build_bsa_network(activation='leaky_relu', neurons=100)
        model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'])
        model.load_weights('/Git/trading-agent/aggregated_agent/networks/bsa.h5f')
        return model

    if model_type == 'boa':
        max_len = util.get_maxlen_of_binary_array(120)
        model = build_boa_network(max_len, neurons=100, activation='leaky_relu')
        model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
        model.load_weights('/Git/trading-agent/aggregated_agent/networks/boa.h5f')
        return model

    if model_type == 'ssa':
        model = build_ssa_network()
        model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'accuracy'])
        model.load_weights('/Git/trading-agent/aggregated_agent/networks/ssa.h5f')
        return model

    if model_type == 'soa':
        max_len = util.get_maxlen_of_binary_array(120)
        model = build_soa_network(max_len, neurons=100, activation='leaky_relu')
        model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
        model.load_weights('/Git/trading-agent/aggregated_agent/networks/soa.h5f')  # << soa weight 경로
        return model


_len_observation = 120


def build_bsa_network(optimizer='adam',init_mode='uniform', filters=16, neurons=20, activation='relu'):
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

    i_concatenated_all_h = Dense(neurons, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

    output = Dense(1, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

    model = Model([input_order, input_tranx], output)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    # model.summary()

    return model


def build_boa_network(max_len=7, init_mode='uniform', neurons=20, activation='relu'):
    if activation == 'leaky_relu':
        input_order = Input(shape=(10, 2, 120, 2), )
        input_tranx = Input(shape=(120, 11), )
        input_left_time = Input(shape=(max_len,), )

        h_conv1d_2 = Conv1D(kernel_initializer=init_mode, filters=16, kernel_size=3)(input_tranx)
        h_conv1d_2 = LeakyReLU(alpha=0.3)(h_conv1d_2)
        h_conv1d_4 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(h_conv1d_2)
        h_conv1d_6 = Conv1D(kernel_initializer=init_mode, filters=32, kernel_size=3)(h_conv1d_4)
        h_conv1d_6 = LeakyReLU(alpha=0.3)(h_conv1d_6)
        h_conv1d_8 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(h_conv1d_6)

        h_conv3d_1_1 = Conv3D(kernel_initializer=init_mode, filters=16, kernel_size=(2, 1, 5))(input_order)
        h_conv3d_1_1 = LeakyReLU(alpha=0.3)(h_conv3d_1_1)
        h_conv3d_1_2 = Conv3D(kernel_initializer=init_mode, filters=16, kernel_size=(1, 2, 5))(input_order)
        h_conv3d_1_2 = LeakyReLU(alpha=0.3)(h_conv3d_1_2)

        h_conv3d_1_3 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_1)
        h_conv3d_1_4 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_2)

        h_conv3d_1_5 = Conv3D(kernel_initializer=init_mode, filters=32, kernel_size=(1, 2, 5))(h_conv3d_1_3)
        h_conv3d_1_5 = LeakyReLU(alpha=0.3)(h_conv3d_1_5)
        h_conv3d_1_6 = Conv3D(kernel_initializer=init_mode, filters=32, kernel_size=(2, 1, 5))(h_conv3d_1_4)
        h_conv3d_1_6 = LeakyReLU(alpha=0.3)(h_conv3d_1_6)

        h_conv3d_1_7 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_5)
        h_conv3d_1_8 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_6)
        o_conv3d_1 = Concatenate(axis=-1)([h_conv3d_1_7, h_conv3d_1_8])

        o_conv3d_1_1 = Flatten()(o_conv3d_1)

        i_concatenated_all_h_1 = Flatten()(h_conv1d_8)

        i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1, input_left_time])

        i_concatenated_all_h = Dense(neurons, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

        output = Dense(1, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

        model = Model([input_order, input_tranx, input_left_time], output)

        return model

    input_order = Input(shape=(10, 2, 120, 2), )
    input_tranx = Input(shape=(120, 11), )
    input_left_time = Input(shape=(max_len,), )

    h_conv1d_2 = Conv1D(kernel_initializer=init_mode, filters=16, kernel_size=3, activation=activation)(input_tranx)
    h_conv1d_4 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(h_conv1d_2)
    h_conv1d_6 = Conv1D(kernel_initializer=init_mode, filters=32, kernel_size=3, activation=activation)(h_conv1d_4)
    h_conv1d_8 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(h_conv1d_6)

    h_conv3d_1_1 = Conv3D(kernel_initializer=init_mode, filters=16, kernel_size=(2, 1, 5), activation=activation)(input_order)
    h_conv3d_1_2 = Conv3D(kernel_initializer=init_mode, filters=16, kernel_size=(1, 2, 5), activation=activation)(input_order)

    h_conv3d_1_3 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_1)
    h_conv3d_1_4 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_2)

    h_conv3d_1_5 = Conv3D(kernel_initializer=init_mode, filters=32, kernel_size=(1, 2, 5), activation=activation)(h_conv3d_1_3)
    h_conv3d_1_6 = Conv3D(kernel_initializer=init_mode, filters=32, kernel_size=(2, 1, 5), activation=activation)(h_conv3d_1_4)

    h_conv3d_1_7 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_5)
    h_conv3d_1_8 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_6)
    o_conv3d_1 = Concatenate(axis=-1)([h_conv3d_1_7, h_conv3d_1_8])

    o_conv3d_1_1 = Flatten()(o_conv3d_1)

    i_concatenated_all_h_1 = Flatten()(h_conv1d_8)

    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1, input_left_time])

    i_concatenated_all_h = Dense(neurons, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

    output = Dense(1, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

    model = Model([input_order, input_tranx, input_left_time], output)

    return model




def build_ssa_network():
    max_len = util.get_maxlen_of_binary_array(120)
    input_order = Input(shape=(10, 2, _len_observation, 2), )
    input_tranx = Input(shape=(_len_observation, 11), )
    input_elapedtime = Input(shape=(max_len,), )
    input_lefttime = Input(shape=(max_len,), )

    h_conv1d_2 = Conv1D(filters=16, kernel_size=3)(input_tranx)
    h_conv1d_2 = LeakyReLU(alpha=0.3)(h_conv1d_2)

    h_conv1d_4 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(h_conv1d_2)

    h_conv1d_6 = Conv1D(filters=32, kernel_size=3)(h_conv1d_4)
    h_conv1d_6 = LeakyReLU(alpha=0.3)(h_conv1d_6)

    h_conv1d_8 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(h_conv1d_6)

    h_conv3d_1_1 = Conv3D(filters=16, kernel_size=(2, 1, 5))(input_order)
    h_conv3d_1_1 = LeakyReLU(alpha=0.3)(h_conv3d_1_1)

    h_conv3d_1_2 = Conv3D(filters=16, kernel_size=(1, 2, 5))(input_order)
    h_conv3d_1_2 = LeakyReLU(alpha=0.3)(h_conv3d_1_2)

    h_conv3d_1_3 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_1)
    h_conv3d_1_4 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_2)

    h_conv3d_1_5 = Conv3D(filters=32, kernel_size=(1, 2, 5))(h_conv3d_1_3)
    h_conv3d_1_5 = LeakyReLU(alpha=0.3)(h_conv3d_1_5)

    h_conv3d_1_6 = Conv3D(filters=32, kernel_size=(2, 1, 5))(h_conv3d_1_4)
    h_conv3d_1_6 = LeakyReLU(alpha=0.3)(h_conv3d_1_6)

    h_conv3d_1_7 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_5)
    h_conv3d_1_8 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_6)
    o_conv3d_1 = Concatenate(axis=-1)([h_conv3d_1_7, h_conv3d_1_8])

    o_conv3d_1_1 = Flatten()(o_conv3d_1)

    i_concatenated_all_h_1 = Flatten()(h_conv1d_8)

    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1, input_elapedtime, input_lefttime])

    hidden_out = Dense(100)(i_concatenated_all_h)
    hidden_out = LeakyReLU(alpha=0.3)(hidden_out)

    output = Dense(1, activation='linear')(hidden_out)

    model = Model([input_order, input_tranx, input_elapedtime, input_lefttime], output)

    return model


def build_soa_network(max_len=7, init_mode='uniform', neurons=20, activation='relu'):
    if activation == 'leaky_relu':
        input_order = Input(shape=(10, 2, 120, 2), )
        input_tranx = Input(shape=(120, 11), )
        input_left_time = Input(shape=(max_len,), )
        elapsed_time = Input(shape=(max_len,), )

        h_conv1d_2 = Conv1D(kernel_initializer=init_mode, filters=16, kernel_size=3)(input_tranx)
        h_conv1d_2 = LeakyReLU(alpha=0.3)(h_conv1d_2)
        h_conv1d_4 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(h_conv1d_2)
        h_conv1d_6 = Conv1D(kernel_initializer=init_mode, filters=32, kernel_size=3)(h_conv1d_4)
        h_conv1d_6 = LeakyReLU(alpha=0.3)(h_conv1d_6)
        h_conv1d_8 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(h_conv1d_6)

        h_conv3d_1_1 = Conv3D(kernel_initializer=init_mode, filters=16, kernel_size=(2, 1, 5))(input_order)
        h_conv3d_1_1 = LeakyReLU(alpha=0.3)(h_conv3d_1_1)
        h_conv3d_1_2 = Conv3D(kernel_initializer=init_mode, filters=16, kernel_size=(1, 2, 5))(input_order)
        h_conv3d_1_2 = LeakyReLU(alpha=0.3)(h_conv3d_1_2)

        h_conv3d_1_3 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_1)
        h_conv3d_1_4 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_2)

        h_conv3d_1_5 = Conv3D(kernel_initializer=init_mode, filters=32, kernel_size=(1, 2, 5))(h_conv3d_1_3)
        h_conv3d_1_5 = LeakyReLU(alpha=0.3)(h_conv3d_1_5)
        h_conv3d_1_6 = Conv3D(kernel_initializer=init_mode, filters=32, kernel_size=(2, 1, 5))(h_conv3d_1_4)
        h_conv3d_1_6 = LeakyReLU(alpha=0.3)(h_conv3d_1_6)

        h_conv3d_1_7 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_5)
        h_conv3d_1_8 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_6)
        o_conv3d_1 = Concatenate(axis=-1)([h_conv3d_1_7, h_conv3d_1_8])

        o_conv3d_1_1 = Flatten()(o_conv3d_1)

        i_concatenated_all_h_1 = Flatten()(h_conv1d_8)

        i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1, input_left_time, elapsed_time])

        i_concatenated_all_h = Dense(neurons, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

        output = Dense(1, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

        model = Model([input_order, input_tranx, input_left_time, elapsed_time], output)

        return model

    input_order = Input(shape=(10, 2, 120, 2), )
    input_tranx = Input(shape=(120, 11), )
    input_left_time = Input(shape=(max_len,), )
    elapsed_time = Input(shape=(max_len,), )

    h_conv1d_2 = Conv1D(kernel_initializer=init_mode, filters=16, kernel_size=3, activation=activation)(input_tranx)
    h_conv1d_4 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(h_conv1d_2)
    h_conv1d_6 = Conv1D(kernel_initializer=init_mode, filters=32, kernel_size=3, activation=activation)(h_conv1d_4)
    h_conv1d_8 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(h_conv1d_6)

    h_conv3d_1_1 = Conv3D(kernel_initializer=init_mode, filters=16, kernel_size=(2, 1, 5), activation=activation)(input_order)
    h_conv3d_1_2 = Conv3D(kernel_initializer=init_mode, filters=16, kernel_size=(1, 2, 5), activation=activation)(input_order)

    h_conv3d_1_3 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_1)
    h_conv3d_1_4 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_2)

    h_conv3d_1_5 = Conv3D(kernel_initializer=init_mode, filters=32, kernel_size=(1, 2, 5), activation=activation)(h_conv3d_1_3)
    h_conv3d_1_6 = Conv3D(kernel_initializer=init_mode, filters=32, kernel_size=(2, 1, 5), activation=activation)(h_conv3d_1_4)

    h_conv3d_1_7 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_5)
    h_conv3d_1_8 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_6)
    o_conv3d_1 = Concatenate(axis=-1)([h_conv3d_1_7, h_conv3d_1_8])

    o_conv3d_1_1 = Flatten()(o_conv3d_1)

    i_concatenated_all_h_1 = Flatten()(h_conv1d_8)

    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1, input_left_time, elapsed_time])

    i_concatenated_all_h = Dense(neurons, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

    output = Dense(1, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

    model = Model([input_order, input_tranx, input_left_time, elapsed_time], output)

    return model
