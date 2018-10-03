
import os
import sys

#newPath = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))) + '/trading-gym'
#sys.path.append(newPath)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-import-gym", "--import-gym",help="import trading gym", action="store_true")
parser.add_argument("-gym-dir", "--gym-dir", type=str, help="import trading gym")
parser.add_argument("-project-dir", "--project-dir", type=str, help="import project home")
parser.add_argument("-model-index", "--model-index", type=int, help="model parameters index")
parser.add_argument("-device", "--device", type=int, help="model parameter")
args = parser.parse_args()

if args.import_gym:
    import sys
    sys.path.insert(0, args.gym_dir)
    sys.path.insert(1, args.project_dir)

import keras.backend as K
from keras.models import Model
from keras.layers import LeakyReLU, Input, Dense, Conv3D, Conv1D, Dense, Flatten, MaxPooling1D, MaxPooling2D,MaxPooling3D,Concatenate
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from gym_core.ioutil import *  # file i/o to load stock csv files
from core.scikit_learn_multi_input_4 import KerasRegressor
from sklearn.model_selection import GridSearchCV

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import logging
import pickle
import config
import sell_signal_agent.ssa_metrics as mt

if args.device is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.SSA_PARAMS['P_TRAINING_GPU'])
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
_len_observation = int(config.SSA_PARAMS['P_OBSERVATION_LEN'])
_pickle_training_dir = config.SSA_PARAMS['PICKLE_DIR_FOR_TRAINING']

if args.model_index is None:
    model_index = 0
else:
    model_index = args.model_index

model_params = [
    {
        'epochs' : 1,
        'batchsize' : 10,
        'neurons' : 50,
        'activation' : 'leaky_relu'
    },
    {
        'epochs' : 75,
        'batchsize' : 70,
        'neurons' : 75,
        'activation' : 'leaky_relu'
    },
    {
        'epochs' : 50,
        'batchsize' : 70,
        'neurons' : 125,
        'activation' : 'leaky_relu'
    },
    {
        'epochs' : 100,
        'batchsize' : 30,
        'neurons' : 175,
        'activation' : 'leaky_relu'
    },
    {
        'epochs' : 100,
        'batchsize' : 50,
        'neurons' : 175,
        'activation' : 'leaky_relu'
    },
    {
        'epochs' : 100,
        'batchsize' : 70,
        'neurons' : 125,
        'activation' : 'leaky_relu'
    }
]

"""
it will prevent process not to occupying 100% of gpu memory for the first time. 
Instead, it will use memory incrementally.
"""
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# set_session(tf.Session(config=config))

"""
build q newtork using cnn and dense layer
"""

"""
build q newtork using cnn and dense layer
"""
def build_network_for_sparsed(optimizer='adam',init_mode='uniform',
                              filters=16, neurons=100, activation='relu', ssa_model_params=model_params):

    if activation == 'leaky_relu':
        activation = LeakyReLU(alpha=0.3)

    neurons = ssa_model_params['neurons']

    input_order = Input(shape=(10, 2, _len_observation, 2), name="x1")
    input_tranx = Input(shape=(_len_observation, 11), name="x2")
    input_elapedtime = Input(shape=(max_len,), name="x3")
    input_lefttime = Input(shape=(max_len,), name="x4")

    h_conv1d_2 = Conv1D(filters=filters, kernel_initializer=init_mode, kernel_size=3)(input_tranx)
    h_conv1d_2 = LeakyReLU(alpha=0.3)(h_conv1d_2)

    h_conv1d_4 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(h_conv1d_2)

    h_conv1d_6 = Conv1D(filters=filters*2, kernel_initializer=init_mode, kernel_size=3)(h_conv1d_4)
    h_conv1d_6 = LeakyReLU(alpha=0.3)(h_conv1d_6)

    h_conv1d_8 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(h_conv1d_6)

    h_conv3d_1_1 = Conv3D(filters=filters, kernel_initializer=init_mode, kernel_size=(2, 1, 5))(input_order)
    h_conv3d_1_1 = LeakyReLU(alpha=0.3)(h_conv3d_1_1)

    h_conv3d_1_2 = Conv3D(filters=filters, kernel_initializer=init_mode, kernel_size=(1, 2, 5))(input_order)
    h_conv3d_1_2 = LeakyReLU(alpha=0.3)(h_conv3d_1_2)

    h_conv3d_1_3 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_1)
    h_conv3d_1_4 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_2)

    h_conv3d_1_5 = Conv3D(filters=filters*2, kernel_initializer=init_mode, kernel_size=(1, 2, 5))(h_conv3d_1_3)
    h_conv3d_1_5 = LeakyReLU(alpha=0.3)(h_conv3d_1_5)

    h_conv3d_1_6 = Conv3D(filters=filters*2, kernel_initializer=init_mode, kernel_size=(2, 1, 5))(h_conv3d_1_4)
    h_conv3d_1_6 = LeakyReLU(alpha=0.3)(h_conv3d_1_6)

    h_conv3d_1_7 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_5)
    h_conv3d_1_8 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_6)
    o_conv3d_1 = Concatenate(axis=-1)([h_conv3d_1_7, h_conv3d_1_8])

    o_conv3d_1_1 = Flatten()(o_conv3d_1)

    i_concatenated_all_h_1 = Flatten()(h_conv1d_8)

    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1, input_elapedtime, input_lefttime])

    i_concatenated_all_h = Dense(neurons, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

    output = Dense(1, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

    model = Model([input_order, input_tranx, input_elapedtime, input_lefttime], output)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', 'mae', 'mape', mt.mean_pred, mt.theil_u, mt.r])
    model.summary()

    return model

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


def get_real_data_sparsed(pickle_dir, ticker='001470', date='20180420', train_data_rows=None):
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
    x3_dimension_info = (max_len,)
    x4_dimension_info = (max_len,)
    # y1_dimension_info = (120,)

    pickle_name = pickle_dir + os.path.sep + current_ticker + '_' + current_date + '.pickle'
    f = open(pickle_name, 'rb')
    d = pickle.load(f)  # d[data_type][second] : mapobject!!
    f.close()

    total_rows = len(d[0])

    if train_data_rows is None:
        train_data_rows = len(d[0])

    x1 = np.zeros([10, 2, _len_observation, 2])
    x2 = np.zeros([_len_observation, 11])
    x3 = np.zeros([max_len,])
    x4 = np.zeros([max_len,])
    y1 = np.zeros([120])

    d_x1 = []
    d_x2 = []
    d_x3 = []
    d_x4 = []
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

        binary_second = seconds_to_binary_array(d[2][idx], max_len)
        for feature in range(x3_dimension_info[0]):  # max_len :features
            x3[feature] = binary_second[feature]
        d_x3.append(x3)

        binary_second = seconds_to_binary_array(d[3][idx], max_len)
        for feature in range(x4_dimension_info[0]):  # max_len :features
            x4[feature] = binary_second[feature]
        d_x4.append(x4)


        # for second in range(y1_dimension_info[0]): # 60 : seconds
        d_y1.append(d[4][idx])

    return np.asarray(d_x1), np.asarray(d_x2), np.asarray(d_x3), np.asarray(d_x4), np.asarray(d_y1)


def train_using_real_data_sparsed(pickle_dir, ssa_model_params=model_params):

    model = build_network_for_sparsed(ssa_model_params=ssa_model_params)

    l = load_ticker_yyyymmdd_list_from_directory(pickle_dir)

    t_x1, t_x2, t_x3, t_x4, t_y1 = [],[],[],[],[]

    for (ti, da) in l:
        print('loading data from ticker {}, yyyymmdd {} is started.'.format(ti, da))
        x1, x2, x3, x4, y1 = load_data_sparsed(ti, da, pickle_dir=pickle_dir, use_fake_data=False)
        t_x1.append(x1)
        t_x2.append(x2)
        t_x3.append(x3)
        t_x4.append(x4)
        t_y1.append(y1)
        print('loading data from ticker {}, yyyymmdd {} is finished.'.format(ti, da))
    t_x1 = np.concatenate(t_x1)
    t_x2 = np.concatenate(t_x2)
    t_x3 = np.concatenate(t_x3)
    t_x4 = np.concatenate(t_x4)

    t_y1 = np.concatenate(t_y1)

    print('total x1 : {}, total x2 : {}, total x3 : {}, total x4 : {}, total y1 : {}'.format(len(t_x1), len(t_x2), len(t_x3), len(t_x4), len(t_y1)))
    # np.append(t_x1,values=(36,8,10,2,60))

    # {steps} --> this file will be saved whenever it runs every steps as much as {step}
    checkpoint_weights_filename = 'ssa_' + 'fill_params_information_in_here' + '_weights_{step}.h5f'

    # TODO: here we can add hyperparameters information like below!!
    log_filename = 'ssa_{}_log.json'.format('fill_params_information_in_here')
    checkpoint_interval = 50

    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=checkpoint_interval)]
    callbacks += [FileLogger(log_filename, interval=100)]

    print('start to train.')

    param_epochs = ssa_model_params['epochs']
    param_batch_size = ssa_model_params['batchsize']
    param_neurons = ssa_model_params['neurons']
    history = model.fit({'x1': t_x1, 'x2': t_x2, 'x3': t_x3, 'x4': t_x4}, t_y1, epochs=param_epochs, verbose=2, batch_size=param_batch_size, callbacks=callbacks)

    name_subfix = '_e' + str(param_epochs) + "_b" + str(param_batch_size) + "_n" + str(param_neurons)

    f = open("ssa_model_history" + name_subfix, 'wb')
    pickle.dump(history.history, f)
    f.close()

    # with open('ssa_model_history', 'wb') as file_pi:
    #     pickle.dump(history.history, file_pi)

    model.save_weights('weight' + name_subfix + '.h5f')
    model.save('model' + name_subfix + '.h5')
    plot_history(history, mt.dict_to_plot, ssa_model_params, 'fig_save')


def load_data_sparsed(t, d, pickle_dir, use_fake_data=False):
    if use_fake_data:
        x1, x2, x3, x4, y = get_sample_data(10)
    else:
        current_date = d
        current_ticker = t
        #if you give second as None, it will read every seconds in file.
        x1, x2, x3, x4, y = get_real_data_sparsed(pickle_dir, current_ticker, current_date)
    return x1, x2, x3, x4, y


def train_using_real_data_sparsed_gs(pickle_dir):

    l = load_ticker_yyyymmdd_list_from_directory(pickle_dir)

    t_x1, t_x2, t_x3, t_x4, t_y1 = [], [], [], [], []

    for (ti, da) in l:
        print('loading data from ticker {}, yyyymmdd {} is started.'.format(ti, da))
        x1, x2, x3, x4, y1 = load_data_sparsed(ti, da, pickle_dir=pickle_dir, use_fake_data=False)
        t_x1.append(x1)
        t_x2.append(x2)
        t_x3.append(x3)
        t_x4.append(x4)
        t_y1.append(y1)
        print('loading data from ticker {}, yyyymmdd {} is finished.'.format(ti, da))
    t_x1 = np.concatenate(t_x1)
    t_x2 = np.concatenate(t_x2)
    t_x3 = np.concatenate(t_x3)
    t_x4 = np.concatenate(t_x4)
    t_y1 = np.concatenate(t_y1)

    print('total x1 : {}, total x2 : {}, total x3 : {}, total x4 : {}, total y1 : {}'.format(len(t_x1), len(t_x2), len(t_x3), len(t_x4), len(t_y1)))

    # {steps} --> this file will be saved whenever it runs every steps as much as {step}
    checkpoint_weights_filename = 'ssa_' + 'fill_params_information_in_here' + '_weights_{step}.h5f'

    # TODO: here we can add hyperparameters information like below!!
    log_filename = 'ssa_{}_log.json'.format('fill_params_information_in_here')
    checkpoint_interval = 50

    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=checkpoint_interval)]
    callbacks += [FileLogger(log_filename, interval=100)]

    print('start to train.')

    # create model
    model = KerasRegressor(build_fn=build_network_for_sparsed, verbose=0)

    """
    define the grid search parameters
    """
    # simple try!!
    # batch_size = [10]
    # epochs = [10]
    # neurons = [20]
    # activation = ['leaky_relu']

    # first try
    # end up with Best: -46695.504027 using {'activation': 'leaky_relu', 'batch_size': 10, 'epochs': 50, 'neurons': 50}
    # batch_size = [10, 20, 40, 100]
    # epochs = [10, 50]
    # neurons = [20, 25, 50]
    # activation = ['relu', 'leaky_relu', 'tanh']

    # todo : second try!
    # batch_size = [10]
    # epochs = [70, 100]
    # neurons = [70, 100]
    # activation = ['leaky_relu']

    # todo : third try!
    batch_size = [10,20,30]
    epochs = [50]
    neurons = [75]
    activation = ['leaky_relu']

    param_grid = dict(batch_size=batch_size, epochs=epochs, neurons=neurons, activation=activation)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

    grid_result = grid.fit(np.array([{'x1': a, 'x2': b, 'x3': c, 'x4': d} for a, b, c, d in zip(t_x1, t_x2, t_x3, t_x4)]), t_y1)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def plot_history(history, to_plot, params, save_path):
    ## params ##
    batch_size = params['batchsize']
    epochs = params['epochs']
    neurons = params['neurons']
    activation = params["activation"]

    for key in to_plot.keys():

        file_name = 'bs' + str(batch_size) + '_ep' + str(epochs) + '_nrs' + str(neurons) + '_act(' + str(activation) + ')_'+ key + '.png'
        category = to_plot[key]
        plt.plot(history.history[category])
        # plt.plot(history.history['val_' + category])
        plt.title(key)
        plt.ylabel(key)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(save_path + '/' + file_name)
        # plt.show()

dat = np.arange(1, 13) / 2.0
def discretize(data, bins):
    split = np.array_split(np.sort(data), bins)
    cutoffs = [x[-1] for x in split]
    cutoffs = cutoffs[:-1]
    discrete = np.digitize(data, cutoffs, right=True)
    return discrete, cutoffs
def get_maxlen_of_binary_array(max_seconds):
    return len(np.binary_repr(max_seconds))
def seconds_to_binary_array(seconds, max_len):
    return np.binary_repr(seconds).zfill(max_len)

max_len = get_maxlen_of_binary_array(120)
train_using_real_data_sparsed(_pickle_training_dir, model_params[model_index])
#train_using_real_data_sparsed_gs(_pickle_training_dir)


