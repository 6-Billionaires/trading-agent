from keras.models import Model
from keras.layers import LeakyReLU, Input, Dense, Conv3D, Conv1D, Dense, Flatten, MaxPooling1D, MaxPooling2D,MaxPooling3D,Concatenate
import numpy as np
import pickle
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from gym_core import ioutil  # file i/o to load stock csv files
import pickle
import os
import config
from gym_core.ioutil import *  # file i/o to load stock csv files
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-import-gym", "--import-gym",help="import trading gym", action="store_true")

args = parser.parse_args()

if args.import_gym:
    import sys
    sys.path.insert(0, config.GYM['HOME'])

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.BSA_PARAMS['P_TRAINING_GPU'])
_len_observation = int(config.BSA_PARAMS['P_OBSERVATION_LEN'])
_pickle_test_dir = config.BSA_PARAMS['PICKLE_DIR_FOR_TEST']


"""
same function as the function in train.py 
"""
def build_network():
    input_order = Input(shape=(10, 2, 120, 2), name="x1")
    input_tranx = Input(shape=(120, 11), name="x2")

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

    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1])

    output = Dense(1, activation='linear')(i_concatenated_all_h)

    model = Model([input_order, input_tranx], output)

    return model


"""
build q newtork using cnn and dense layer
"""
"""
build q newtork using cnn and dense layer
"""
def build_network_for_sparsed(optimizer='adam',init_mode='uniform',
                              filters=16, neurons=20, activation='relu'):
    if activation == 'leaky_relu':
        activation = LeakyReLU(alpha=0.3)

    input_order = Input(shape=(10, 2, _len_observation, 2), name="x1")
    input_tranx = Input(shape=(_len_observation, 11), name="x2")

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



def get_real_data_sparsed(dir, ticker='001470', date='20180420', train_data_rows=None):
    """
    Get sparsed data for supervised learning
    :param dir : directory where pickle files exist
    :param ticker: ticker number to read
    :param date: date yyyymmdd to read
    :param train_data_rows: data rows to read for training, default None : read all rows
    :param save_dir: default ''
    :return: training data x1, x2, y1 for supervised learning model
    """
    current_ticker = ticker
    current_date = date

    x1_dimension_info = (10, 2, _len_observation, 2)  # 60 --> 120 (@iljoo)
    x2_dimension_info = (_len_observation, 11)
    # y1_dimension_info = (120,)

    pickle_name = dir + os.path.sep + current_ticker + '_' + current_date + '.pickle'
    f = open(pickle_name, 'rb')
    d = pickle.load(f)  # d[data_type][second] : mapobject!!
    f.close()

    total_rows = len(d[0])

    if train_data_rows is None:
        train_data_rows = len(d[0])

    x1 = np.zeros([10, 2, _len_observation, 2])
    x2 = np.zeros([_len_observation, 11])
    y1 = np.zeros([120])

    d_x1 = []
    d_x2 = []
    d_y1 = []

    for idx in range(train_data_rows):
        for row in range(x1_dimension_info[0]):  # 10 : row
            for column in range(x1_dimension_info[1]):  # 2 : column
                for second in range(x1_dimension_info[2]):  # 60: seconds
                    for channel in range(x1_dimension_info[3]):  # 2 : channel
                        x1[row][column][second][channel] = d[0][idx][second][channel*20+column*10]
        d_x1.append(x1)

        for second in range(x2_dimension_info[0]):  # 120 : seconds
            for feature in range(x2_dimension_info[1]):  # 11 : features
                x2[second, feature] = d[1][idx][second][feature]
        d_x2.append(x2)

        # for second in range(y1_dimension_info[0]): # 60 : seconds
        d_y1.append(d[2][idx])

    return np.asarray(d_x1), np.asarray(d_x2), np.asarray(d_y1)



def get_sample_sparsed_data(count):
    start = 0
    ld_x1 = []
    ld_x2 = []
    ld_y = []
    d1 = []

    # x1_dimension_info = (10, 2, 120, 2)  # 60 --> 120 (@ilzoo)
    # x2_dimension_info = (120, 11)
    # y1_dimension_info = (120,)

    for i in range(count):
        d1_2 = np.arange(start, start + 2 * 10 * _len_observation * 2).reshape([10, 2, _len_observation, 2])
        start += 2 * 10 * _len_observation
        d2 = np.arange(start, start+11*60).reshape([60, 11])
        start += 11 * _len_observation
        ld_x1.append(d1_2)

        # ld_x1.append([d1_1, d1_2])
        ld_x2.append(d2)

    for j in range(count):
        d1 = np.arange(start, start + 1)
        ld_y.append(d1)

    return np.asarray(ld_x1), np.asarray(ld_x2), np.asarray(ld_y)



def get_sample_data(count):
    start = 0
    ld_x1 = []
    ld_x2 = []
    ld_y = []
    d1 = []

    # x1_dimension_info = (10, 2, 120, 2)  # 60 --> 120 (@ilzoo)
    # x2_dimension_info = (120, 11)
    # y1_dimension_info = (120,)

    for i in range(count):
        d1_2 = np.arange(start, start + 2 * 10 * 120 * 2).reshape([10, 2, 120, 2])
        start += 2 * 10 * 120
        d2 = np.arange(start, start+11*120).reshape([120, 11])
        start += 11 * 120
        ld_x1.append(d1_2)

        # ld_x1.append([d1_1, d1_2])
        ld_x2.append(d2)

    for j in range(count):
        d1 = np.arange(start, start + 1)
        ld_y.append(d1)

    return np.asarray(ld_x1), np.asarray(ld_x2), np.asarray(ld_y)




@runtime
def load_data_sparsed(t, d, dir, use_fake_data=False):
    if use_fake_data:
        x1, x2, y = get_sample_data(10)
    else:
        current_date = d
        current_ticker = t
        #if you give second as None, it will read every seconds in file.
        # x1, x2, y = get_real_data(current_ticker, current_date, train_data_rows=130)
        x1, x2, y = get_real_data_sparsed(dir, current_ticker, current_date)
    return x1, x2, y


model = build_network_for_sparsed(activation='leaky_relu', neurons=100)
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape', 'mse'])
model.summary()

# load weight
model.load_weights('bsa_weight.h5f')

# x1, x2, y = get_sample_sparsed_data(10) # for temporary test

l = load_ticker_yyyymmdd_list_from_directory(_pickle_test_dir)
t_x1, t_x2, t_y1 = [], [], []

for (ti, da) in l:
    print('loading data from ticker {}, yyyymmdd {} is started.'.format(ti, da))
    x1, x2, y1 = load_data_sparsed(ti, da, dir=_pickle_test_dir, use_fake_data=False)
    t_x1.append(x1)
    t_x2.append(x2)
    t_y1.append(y1)
    print('loading data from ticker {}, yyyymmdd {} is finished.'.format(ti, da))
t_x1 = np.concatenate(t_x1)
t_x2 = np.concatenate(t_x2)
t_y1 = np.concatenate(t_y1)

print('total x1 : {}, total x2 : {}, total y1 : {}'.format(len(t_x1), len(t_x2), len(t_y1)))

scores = model.evaluate({'x1': t_x1, 'x2': t_x2}, t_y1, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

with open('bsa_evaluate_model_history_1809142003', 'wb') as file_pi:
    pickle.dump(scores, file_pi)