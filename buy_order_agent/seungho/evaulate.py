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
# from core.scikit_learn_multi_input import KerasRegressor
# from sklearn.model_selection import GridSearchCV

"""
build q newtork using cnn and dense layer
"""


def build_network(max_secs=90, max_len=7, optimizer='adam', init_mode='uniform', ):
    input_order = Input(shape=(10, 2, max_secs, 2), name="x1")
    input_tranx = Input(shape=(max_secs, 11), name="x2")
    input_remain_secs = Input(shape=(max_len,), name="x3")  # update. remained seconds up to 180 seconds ??

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

    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1, input_remain_secs])

    i_concatenated_all_h = Dense(10, activation='linear')(i_concatenated_all_h)
    output = Dense(1, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

    model = Model([input_order, input_tranx, input_remain_secs], output)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary()

    return model


def get_real_data(max_secs, max_len, csv, pickles, max_stock = 10):
    x1_dimension_info = (10, 2, max_secs, 2)
    x2_dimension_info = (max_secs, 11)
    x3_dimension_info = (max_len,)

    d_x1 = []
    d_x2 = []
    d_x3 = []
    d_y1 = []

    keys = list(pickles.keys())
    keys = keys[::-1]
    # random.shuffle(keys)

    stock = 0
    for key in keys:
        stock = stock + 1
        if stock > max_stock:
            break
        sys.stdout.write("\r%i" %stock + " / %i 완료"  %max_stock)
        sys.stdout.flush()

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

        x3 = np.zeros([max_len])
        binary_second = util.seconds_to_binary_array(max_secs, max_len)
        for feature in range(x3_dimension_info[0]):  # max_len :features
            x3[feature] = binary_second[feature]

        d_x3.append(x3)

        # for second in range(y1_dimension_info[0]): #60 : seconds
        d_y1.append(pickles[key][2][start_sec])

    sys.stdout.write("\r")
    sys.stdout.flush()

    return np.asarray(d_x1), np.asarray(d_x2), np.asarray(d_x3), np.asarray(d_y1)


def train_using_real_data(data, max_secs, max_len, temp=True):
    # {일자 + 종목코드} : {meta, quote, order}
    csv = ioutil.load_data_from_directory2('0')
    # {일자 + 종목코드} : [second, left_time, elapsed_time, y]
    pickles, max_size = ioutil.load_ticker_yyyymmdd_list_from_directory2(data)

    model = build_network(max_secs, max_len)

    # load weight
    model.load_weights('final_weight.h5')

    x1, x2, x3, y = get_real_data(max_secs, max_len, csv, pickles, 10)

    scores = model.evaluate({'x1': x1, 'x2': x2, 'x3': x3}, y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))



def main():
    # train_using_fake_data()
    # picke path
    directory = os.path.abspath(ioutil.make_dir(os.path.dirname(os.path.abspath(__file__)), 'pickles'))
    # max length of bit for 90
    max_secs = 90
    max_len = util.get_maxlen_of_binary_array(max_secs)
    train_using_real_data(directory, max_secs, max_len)


main()