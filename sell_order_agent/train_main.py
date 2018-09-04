import os
import sys
newPath = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))) + '\\trading-gym'
sys.path.append(newPath)

from gym_core import ioutil  # file i/o to load stock csv files
from keras.models import Model
from keras.layers import Input, Dense, Conv3D, Conv1D, Dense, Flatten, MaxPooling1D, MaxPooling2D,MaxPooling3D,Concatenate
import numpy as np
import pickle
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from core import util
import random

"""
build q newtork using cnn and dense layer
"""
def build_network(max_len):
    input_order = Input(shape=(10, 2, 120, 2), name="x1")
    input_tranx = Input(shape=(120, 11), name="x2")
    input_left_time = Input(shape=(120, max_len), name="x3")
    elapsed_time = Input(shape=(120, max_len), name="x4")

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
    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1, Flatten()(input_left_time), Flatten()(elapsed_time)])

    output = Dense(1, activation='linear')(i_concatenated_all_h)

    model = Model([input_order, input_tranx, input_left_time, elapsed_time], output)

    return model

def get_real_data(max_len, csv, pickles, str_episode, end_episode, train_all_periods=None):
    '''
    left_secs : SSA 에서 신호를 보낼때 남은 시간
    elapsed_secs : SSA 에서 신호를 보낸 후 경과 시간
    최초 pickle 을 생성할 때, left_secs 은 랜덤 생성 하고 elapsed_secs 를 0 ~ left_secs 만큼 생성 했었는데, 데이터가 너무 많아서 랜덤하게 30% 데이터만 생성하도록 하였음. (if random.random() > 0.3: continue)
    한번 학습 시에 모든 종목에 대해 40개의 pickle 을 뽑아서 1개의 episode 를 구성함. 시간 순서를 random 으로 뽑지는 않음. (종목 수 36 * 피클 데이터 수 40 = 1440)
    :param max_len:
    :param csv:
    :param pickles:
    :param str_episode:
    :param end_episode:
    :param train_all_periods:
    :return:
    '''

    x1_dimension_info = (10, 2, 120, 2)  # 60 --> 120 (@ilzoo)
    x2_dimension_info = (120, 11)
    x3_dimension_info = (120, max_len)
    x4_dimension_info = (120, max_len)
    #y1_dimension_info = (120,)

    d_x1 = []
    d_x2 = []
    d_x3 = []
    d_x4 = []
    d_y1 = []

    keys = list(pickles.keys())

    random.shuffle(keys)

    print(str_episode, end_episode, len(pickles), (end_episode - str_episode) * len(pickles))

    for idx in range(str_episode, end_episode):
        sys.stdout.write("\r%i" % idx + " / %i 완료"  %end_episode)
        sys.stdout.flush()
        for key in keys:
            if len(pickles[key]) < idx:
                continue

            # index (second) : second - 120 + i
            #pickles[key][idx][0] - 120
            # left_secs : 120 초간 동일
            #pickles[key][idx][1]
            # elapsed_secs : max(elapsed_secs - 120 + i, 0)
            #pickles[key][idx][2]
            # y
            #pickles[key][idx][3]

            x1 = np.zeros([10,2,120,2])
            for second in range(x1_dimension_info[2]):  # 120 : seconds
                tmp = csv[key]['order'].loc[pickles[key][idx][0]-120+second]
                for row in range(x1_dimension_info[0]):  #10 : row
                    for column in range(x1_dimension_info[1]):  #2 : column
                        for channel in range(x1_dimension_info[3]):  #2 : channel
                            buy_sell = ''
                            if channel == 1:
                                buy_sell = 'Buy'
                            else:
                                buy_sell = 'Sell'

                            if column  == 0:
                                value = 'Hoga'
                            else:
                                value = 'Order'

                            x1[row][column][second][channel] = tmp[buy_sell+value+str(row+1)]
            d_x1.append(x1)

            x2 = np.zeros([120,11])
            for second in range(x2_dimension_info[0]):  #120 : seconds
                tmp = csv[key]['quote'].loc[pickles[key][idx][0]-120+second]
                for feature in range(x2_dimension_info[1]):  #11 :features
                    x2[second, feature] = tmp[feature]
            d_x2.append(x2)

            x3 = np.zeros([120, max_len])
            for second in range(x3_dimension_info[0]):  # 120 : seconds
                binarySecond = util.seconds_to_binary_array(pickles[key][idx][1], max_len)
                for feature in range(x3_dimension_info[1]):  # max_len :features
                    x3[second] = binarySecond[feature]

            d_x3.append(x3)

            x4 = np.zeros([120, max_len])
            for second in range(x4_dimension_info[0]):  # 120 : seconds
                binarySecond = util.seconds_to_binary_array(max(pickles[key][idx][2] - 120 + second, 0), max_len)
                for feature in range(x4_dimension_info[1]):  # max_len :features
                    x4[second] = binarySecond[feature]

            d_x4.append(x4)

            # for second in range(y1_dimension_info[0]): #60 : seconds
            d_y1.append(pickles[key][idx][3])

    sys.stdout.write("\r")
    sys.stdout.flush()
    return np.asarray(d_x1), np.asarray(d_x2), np.asarray(d_x3), np.asarray(d_x4), np.asarray(d_y1)



def train_using_fake_data():
    train_per_each_episode('','',True)

def train_using_real_data(d, max_len):
    # {일자 + 종목코드} : {meta, quote, order}
    csv = ioutil.load_data_from_directory2('0')
    # {일자 + 종목코드} : [second, left_time, elapsed_time, y]
    pickles, max_size = ioutil.load_ticker_yyyymmdd_list_from_directory2(d)

    train_per_each_episode(max_len, csv, pickles, max_size)

def train_per_each_episode(max_len, csv, pickles, max_size):

    # x1, x2, y = get_real_data(current_date,current_ticker,100)
    #if you give second as None, it will read every seconds in file.
    #x1, x2, x3, x4, y = get_real_data(current_ticker, current_date, train_all_periods=130)

    model = build_network(max_len)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()

    # {steps} --> this file will be saved whenver it runs every steps as much as {step}
    checkpoint_weights_filename = 'soa_weights_{step}.h5f'

    #model.load_weights(filepath = checkpoint_weights_filename.format(step=0), by_name=True, skip_mismatch=True)

    # TODO: here we can add hyperparameters information like below!!
    log_filename = 'soa_{}_log.json'.format('fill_params_information_in_here')
    checkpoint_interval = 50000

    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=checkpoint_interval)]
    callbacks += [FileLogger(log_filename, interval=100)]

    # 전체를 몇 episode 로 할 것인가
    max_episode_cnt = 10000
    num_of_episode = int(max_size / max_episode_cnt)

    for episode in range(0, max_episode_cnt):
        x1, x2, x3, x4, y = get_real_data(max_len, csv, pickles, episode * num_of_episode, (episode + 1) * num_of_episode)

        model.fit({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4}, y, epochs=5, verbose=2, batch_size=64, callbacks=callbacks)

        if episode % 10 == 9:
            model.save_weights(filepath=checkpoint_weights_filename.format(step=episode))

    model.save_weights(filepath=checkpoint_weights_filename.format(step='end'))

# train_using_fake_data()
# picke path
directory = os.path.abspath(ioutil.make_dir(os.path.dirname(os.path.abspath(__file__)), 'pickles'))
# max length of bit for 120
max_len = util.get_maxlen_of_binary_array(120)
train_using_real_data(directory, max_len)
