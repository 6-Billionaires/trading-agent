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

"""
build q newtork using cnn and dense layer
"""
def build_network():
    input_order = Input(shape=(10, 2, 120, 2), name="x1")
    input_tranx = Input(shape=(120, 11), name="x2")
    input_elapedtime = Input(shape=(120, 1), name="x3")
    input_lefttime = Input(shape=(120, 1), name="x4")

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
    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1, Flatten()(input_elapedtime), Flatten()(input_lefttime)])

    output = Dense(1, activation='linear')(i_concatenated_all_h)

    model = Model([input_order, input_tranx, input_elapedtime, input_lefttime], output)

    return model

def get_sample_data(count):
    start = 0
    ld_x1 = []
    ld_x2 = []
    ld_x3 = []
    ld_x4 = []
    ld_y = []
    d1 = []

    # x1_dimension_info = (10, 2, 120, 2)  # 60 --> 120 (@ilzoo)
    # x2_dimension_info = (120, 11)
    # y1_dimension_info = (120,)

    for i in range(count):
        d1 = np.arange(start, start + 2 * 10 * 120 * 2).reshape([10, 2, 120, 2])
        start += 2 * 10 * 120
        ld_x1.append(d1)

        d2 = np.arange(start, start + 11 * 120).reshape([120, 11])
        start += 11 * 120
        ld_x2.append(d2)

        d3 = np.arange(start, start + 1 * 120).reshape([120, 1])
        start += 1 * 120
        ld_x3.append(d3)

        d4 = np.arange(start, start + 11 * 120).reshape([120, 11])
        start += 11 * 120
        ld_x4.append(d4)

    for j in range(count):
        d1 = np.arange(start, start + 1)
        ld_y.append(d1)

    return np.asarray(ld_x1), np.asarray(ld_x2), np.asarray(ld_x3), np.asarray(ld_x4), np.asarray(ld_y)

def get_real_data(ticker='001470', date='20180420', train_all_periods=None):

    current_ticker = ticker
    current_date = date

    x1_dimension_info = (10, 2, 120, 2)  # 60 --> 120 (@ilzoo)
    x2_dimension_info = (120, 11)
    x3_dimension_info = (120, 1)
    x4_dimension_info = (120, 1)
    y1_dimension_info = (120,)

    pickle_name = current_ticker + '_' + current_date + '.pickle'
    f = open('./pickles/'+pickle_name, 'rb')
    d = pickle.load(f)  # d[data_type][second] : mapobject!!
    f.close()

    if train_all_periods is None:
        train_all_periods = len(d[0])

    x1 = np.zeros([10,2,120,2])
    x2 = np.zeros([120,11])
    x3 = np.zeros([120, 1])
    x4 = np.zeros([120, 1])
    y1 = np.zeros([120])

    d_x1 = []
    d_x2 = []
    d_x3 = []
    d_x4 = []
    d_y1 = []

    for idx_second in range(train_all_periods):
        if idx_second + 120 > train_all_periods:
            break

        np.zeros([10,2,120,2])
        for row in range(x1_dimension_info[0]):  #10 : row
            for column in range(x1_dimension_info[1]):  #2 : column
                for second in range(x1_dimension_info[2]):  #120 : seconds
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

                        x1[row][column][second][channel] = d[0][idx_second+second][key+value+str(row+1)]
        d_x1.append(x1)

        np.zeros([120,11])
        for second in range(x2_dimension_info[0]):  #120 : seconds
            for feature in range(x2_dimension_info[1]):  #11 :features
                x2[second, feature] = d[1][idx_second+second][feature]
        d_x2.append(x2)

        np.zeros([120, 1])
        for second in range(x3_dimension_info[0]):  # 120 : seconds
            x3[second, 0] = d[2][idx_second]
        d_x3.append(x3)

        np.zeros([120, 1])
        for second in range(x4_dimension_info[0]):  # 120 : seconds
            x3[second, 0] = d[3][idx_second]
        d_x4.append(x4)

        # for second in range(y1_dimension_info[0]): #60 : seconds
        d_y1.append(d[4][idx_second])
    return np.asarray(d_x1), np.asarray(d_x2), np.asarray(d_x3), np.asarray(d_x4),np.asarray(d_y1)



def train_using_fake_data():
    train_per_each_episode('','',True)

def train_using_real_data(d):
    l = ioutil.load_ticker_yyyymmdd_list_from_directory(d)
    for (t,d) in l:
        print('ticker {}, yyyymmdd {} is started for training!'.format(t,d))
        train_per_each_episode(t,d)
        print('ticker {}, yyyymmdd {} is finished for training!'.format(t,d))

def train_per_each_episode(t,d,use_fake_data=False):

    if use_fake_data:
        x1, x2, x3, x4, y = get_sample_data(10)
    else:
        current_date = d
        current_ticker = t

        # x1, x2, y = get_real_data(current_date,current_ticker,100)
        #if you give second as None, it will read every seconds in file.
        # x1, x2, x3, x4, y = get_real_data(current_ticker, current_date, train_all_periods=130)
        x1, x2, x3, x4, y = get_real_data(current_ticker, current_date)

    model = build_network()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()

    # {steps} --> this file will be saved whenver it runs every steps as much as {step}
    checkpoint_weights_filename = 'ssa_' + 'fill_params_information_in_here' + '_weights_{step}.h5f'

    # TODO: here we can add hyperparameters information like below!!
    log_filename = 'ssa_{}_log.json'.format('fill_params_information_in_here')
    checkpoint_interval = 50000

    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=checkpoint_interval)]
    callbacks += [FileLogger(log_filename, interval=100)]

    print('start to train.')
    model.fit({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4}, y, epochs=5, verbose=2, batch_size=64, callbacks=callbacks)
    model.save_weights('final_weight.h5f')

# train_using_fake_data()
d = os.path.abspath(ioutil.make_dir(os.path.dirname(__file__), 'pickles'))
train_using_real_data(d)
