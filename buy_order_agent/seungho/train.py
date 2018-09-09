import os, sys
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


def build_network(max_secs, max_len):
    """
    here, seungho!!

    you need to use functions from core.util package

    first, call get_maxlen_of_binary_array(max_seconds) to find out max length of input size.
    seconds, below function make input data for remaining seconds directly feeding into model
        def seconds_to_binary_array(seconds, max_len):

    :param feature:
    :return:
    """
    input_order = Input(shape=(10, 2, max_secs, 2), name="x1")
    input_tranx = Input(shape=(max_secs, 11), name="x2")
    # input_elapsed_secs = Input(shape=(max_len, ), name="x3") # update. remained seconds up to 180 seconds ??

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

    # i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1, input_elapsed_secs])
    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1])

    i_concatenated_all_h = Dense(10, activation='linear')(i_concatenated_all_h)
    output = Dense(1, activation='linear')(i_concatenated_all_h)

    # model = Model([input_order, input_tranx, input_elapsed_secs], output)
    model = Model([input_order, input_tranx], output)

    return model


def get_real_data(max_secs, max_len, csv, pickles, max_stock = 5):
    x1_dimension_info = (10, 2, max_secs, 2)
    x2_dimension_info = (max_secs, 11)
    # x3_dimension_info = (max_len,)

    d_x1 = []
    d_x2 = []
    # d_x3 = []
    d_y1 = []

    keys = list(pickles.keys())

    random.shuffle(keys)
    # print('------', str_episode, end_episode)
    for stock in range(1, max_stock):

        sys.stdout.write("\r%i" % stock + " / %i 완료"  %max_stock)
        sys.stdout.flush()
        for key in keys:
            # print('------1', episode, key)
            if len(pickles) < stock:
                continue

            # index (second) : second - 120 + i
            #pickles[key][idx][0] - 120
            # left_secs : 120 초간 동일
            #pickles[key][idx][1]
            # elapsed_secs : max(elapsed_secs - 120 + i, 0)
            #pickles[key][idx][2]
            # y
            #pickles[key][idx][3]

            start_sec = 0
            # print('++++', key, episode, pickles[key][0], len(csv[key]['order']))
            x1 = np.zeros([10, 2, max_secs, 2])
            for second in range(x1_dimension_info[2]):  # 90 : seconds

                tmp = csv[key]['order'].loc[pickles[key][0][start_sec]+second]
                # print('------2', second)
                for row in range(x1_dimension_info[0]):  #10 : row
                    for column in range(x1_dimension_info[1]):  #2 : column
                        for channel in range(x1_dimension_info[3]):  #2 : channel
                            buy_sell = ''
                            if channel == 1:
                                buy_sell = 'Buy'
                            else:
                                buy_sell = 'Sell'

                            if column == 0:
                                value = 'Hoga'
                            else:
                                value = 'Order'

                            x1[row][column][second][channel] = tmp[buy_sell+value+str(row+1)]
            d_x1.append(x1)

            x2 = np.zeros([max_secs, 11])
            for second in range(x2_dimension_info[0]):  #120 : seconds
                tmp = csv[key]['quote'].loc[pickles[key][0][start_sec]+second]
                for feature in range(x2_dimension_info[1]):  #11 :features
                    x2[second, feature] = tmp[feature]
            d_x2.append(x2)

            # for second in range(y1_dimension_info[0]): #60 : seconds
            d_y1.append(pickles[key][2][start_sec])

    sys.stdout.write("\r")
    sys.stdout.flush()

    return np.asarray(d_x1), np.asarray(d_x2), np.asarray(d_y1)


def train_using_real_data(data, max_secs, max_len):
    # {일자 + 종목코드} : {meta, quote, order}
    csv = ioutil.load_data_from_directory2('0')
    # {일자 + 종목코드} : [second, left_time, elapsed_time, y]
    pickles, max_size = ioutil.load_ticker_yyyymmdd_list_from_directory2(data)

    train_per_each_episode(max_secs, max_len, csv, pickles, max_size)


def train_per_each_episode(max_secs, max_len, csv, pickles, max_size):

    model = build_network(max_secs, max_len)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()

    # {steps} --> this file will be saved whenver it runs every steps as much as {step}
    checkpoint_weights_filename = 'boa_weights_{step}.h5f'

    #model.load_weights(filepath = checkpoint_weights_filename.format(step=0), by_name=True, skip_mismatch=True)

    # TODO: here we can add hyperparameters information like below!!
    log_filename = 'boa_{}_log.json'.format(max_secs)
    checkpoint_interval = 50000

    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=checkpoint_interval)]
    callbacks += [FileLogger(log_filename, interval=100)]

    # 전체를 몇 episode 로 할 것인가
    max_episode_cnt = 5000
    # num_of_episode = int(max_size / max_episode_cnt)

    for episode in range(0, max_episode_cnt):
        x1, x2, y = get_real_data(max_secs, max_len, csv, pickles)

        print('shape : ', x1.shape, x2.shape, y.shape)
        # model.fit({'x1': x1, 'x2': x2, 'x3': x3}, y, epochs=10, verbose=2, batch_size=64, callbacks=callbacks)
        model.fit({'x1': x1, 'x2': x2}, y, epochs=10, verbose=2, batch_size=64, callbacks=callbacks)

        if episode % 50 == 0:
            model.save_weights(filepath=checkpoint_weights_filename.format(step=episode))

    model.save_weights(filepath=checkpoint_weights_filename.format(step='end'))


def main():
    # train_using_fake_data()
    # picke path
    directory = os.path.abspath(ioutil.make_dir(os.path.dirname(os.path.abspath(__file__)), 'pickles'))
    # max length of bit for 90
    max_secs = 90
    max_len = util.get_maxlen_of_binary_array(max_secs)
    train_using_real_data(directory, max_secs, max_len)


main()