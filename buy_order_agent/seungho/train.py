import os
from keras.models import Model
from keras.layers import Input, Dense, Conv3D, Conv1D, Dense, Flatten, MaxPooling1D, MaxPooling2D,MaxPooling3D,Concatenate
from keras.utils import to_categorical
import numpy as np
import pickle
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from gym_core import ioutil  # file i/o to load stock csv files
import random
from core import util

"""
build q newtork using cnn and dense layer
"""


def build_network():
    """
    here, seungho!!

    you need to use functions from core.util package

    first, call get_maxlen_of_binary_array(max_seconds) to find out max length of input size.
    seconds, below function make input data for remaining seconds directly feeding into model
        def seconds_to_binary_array(seconds, max_len):

    :param feature:
    :return:
    """
    input_order = Input(shape=(10, 2, 90, 2), name="x1")
    input_tranx = Input(shape=(90, 11), name="x2")
    input_remained_secs = Input(shape=(7), name="x3") # update. remained seconds up to 180 seconds ??

    h_conv1d_2 = Conv1D(filters=16, kernel_size=3, activation='relu')(input_tranx)
    h_conv1d_4 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(h_conv1d_2)
    h_conv1d_6 = Conv1D(filters=32, kernel_size=3, activation='relu')(h_conv1d_4)
    h_conv1d_8 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(h_conv1d_6)

    h_conv3d_1_1 = Conv3D(filters=16, kernel_size=(2, 1, 5), activation='relu')(input_order)
    h_conv3d_1_2 = Conv3D(filters=16, kernel_size=(1, 2, 5), activation='relu')(input_order)

    h_conv3d_1_3 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_1)
    h_conv3d_1_4 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_2)

    h_conv3d_1_5 = Conv3D(filters=32, kernel_size=(1, 2, 5), activation='relu')(h_conv3d_1_3)
    h_conv3d_1_6 = Conv3D(filters=32, kernel_size=(2, 1, 5), activation='relu')(h_conv3d_1_4)

    h_conv3d_1_7 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_5)
    h_conv3d_1_8 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_6)
    o_conv3d_1 = Concatenate(axis=-1)([h_conv3d_1_7, h_conv3d_1_8])

    o_conv3d_1_1 = Flatten()(o_conv3d_1)

    i_concatenated_all_h_1 = Flatten()(h_conv1d_8)

    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1, input_remained_secs]) # update. remaining seconds will be concatenated with original data

    i_concatenated_all_h = Dense(10, activation='linear')(i_concatenated_all_h)
    output = Dense(1, activation='linear')(i_concatenated_all_h)

    model = Model([input_order, input_tranx,input_remained_secs], output) # update. remaining seconds will be added into input parameters

    return model


def get_sample_data(feature=5, count=2):
    ld_x1 = []
    ld_x2 = []
    ld_y = []

    for i in range(count):
        d1 = np.arange(0, feature * 90)
        ld_x1.append(d1)

        # y = np.array(y, dtype='int')
        d2 = np.arange(90, dtype='int')
        for j in range(90):
            d2[j] = random.randint(-10, 10)

        encoded = to_categorical(d2)
        print(encoded)
        ld_x2.append(encoded)

    for j in range(count):
        d1 = np.arange(90)
        ld_y.append(d1)

    # print(ld_x1, ld_x2, ld_y)

    return np.asarray(ld_x1), np.asarray(ld_x2), np.asarray(ld_y), feature


def get_real_data(ticker='001470', date='20180420', train_all_periods=None):

    current_ticker = ticker
    current_date = date

    x1_dimension_info = (10, 2, 90, 2)  # 60 --> 90 (@ilzoo)
    x2_dimension_info = (90, 11)
    y1_dimension_info = (90,)

    pickle_name = current_ticker + '_' + current_date + '.pickle'
    f = open(directory + '/' + pickle_name, 'rb')
    data = pickle.load(f)  # d[data_type][second] : mapobject!!
    f.close()

    if train_all_periods is None:
        train_all_periods = len(data[0])

    x1 = np.zeros([10, 2, 90, 2])
    x2 = np.zeros([90, 11])
    x3 = np.zeros([90])
    y1 = np.zeros([90])

    d_x1 = []
    d_x2 = []
    d_x3 = []
    d_y1 = []

    for idx_second in range(train_all_periods):
        if idx_second + 90 > train_all_periods:
            break
        np.zeros([10, 2, 90, 2])
        for row in range(x1_dimension_info[0]):  #10 : row
            for column in range(x1_dimension_info[1]):  #2 : column
                for second in range(x1_dimension_info[2]):  #90 : seconds
                    for channel in range(x1_dimension_info[3]):  #2 : channel
                        key = ''
                        if channel == 1:
                            key = 'Buy'
                        else:
                            key = 'Sell'

                        if column  == 0:
                            value = 'Hoga'
                        else:
                            value = 'Order'

                        x1[row][column][second][channel] = data[0][idx_second+second][key+value+str(row+1)]
        d_x1.append(x1)

        np.zeros([90, 11])
        for second in range(x2_dimension_info[0]):  #90 : seconds
            for feature in range(x2_dimension_info[1]):  #11 :features
                x2[second, feature] = data[1][idx_second+second][feature]
        d_x2.append(x2)
        elapsed_time = data[2][idx_second]
        max_secs = 90
        remained_time = max_secs - elapsed_time.total_seconds()
        binary = '{0:07b}'.format(int(remained_time))
        binary_array = [int(d) for d in str(binary)]
        # d_x3.append(binary) # 그냥 이진수로 넣어도 안되고
        d_x3.append(binary_array) # 이진수를 배열로 넣어도 안되고...

        # for second in range(y1_dimension_info[0]): #60 : seconds
        d_y1.append(data[3][idx_second])

    return np.asarray(d_x1), np.asarray(d_x2), np.asarray(d_x3), np.asarray(d_y1)


def train_using_fake_data():
    train_per_each_episode('', '', True)


def train_using_real_data(directory):
    load = ioutil.load_ticker_yyyymmdd_list_from_directory(directory)
    for (ticker, date) in load:
        print('ticker {}, yyyymmdd {} is started for training!'.format(ticker, date))
        train_per_each_episode(ticker, date)
        print('ticker {}, yyyymmdd {} is finished for training!'.format(ticker, date))


def train_per_each_episode(ticker, date, use_fake_data=False):

    if use_fake_data:
        x1, x2, x_min_secs, y = get_sample_data(10)
    else:
        current_ticker = ticker
        current_date = date

        # x1, x2, y = get_real_data(current_date,current_ticker,100)
        # if you give second as None, it will read every seconds in file.
        x1, x2, x3, y = get_real_data(current_ticker, current_date, train_all_periods=90)

    model = build_network()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()

    # {steps} --> this file will be saved whenver it runs every steps as much as {step}
    checkpoint_weights_filename = 'boa_' + 'fill_params_information_in_here' + '_weights_{step}.h5f'

    # TODO: here we can add hyperparameters information like below!!
    log_filename = 'boa_{}_log.json'.format('fill_params_information_in_here')
    checkpoint_interval = 50000

    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=checkpoint_interval)]
    callbacks += [FileLogger(log_filename, interval=100)]

    print('start to train.')
    model.fit({'x1': x1, 'x2': x2, 'x3': x3}, y, epochs=5, verbose=2, batch_size=64, callbacks=callbacks)


# train_using_fake_data()
directory = os.path.abspath(ioutil.make_dir(os.path.dirname(__file__), 'pickles'))
train_using_real_data(directory)
