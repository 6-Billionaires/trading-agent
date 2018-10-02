from gym_core.ioutil import *  # file i/o to load stock csv files
from keras import metrics
from keras.models import Model
from keras.layers import LeakyReLU, Input, Conv3D, Conv1D, Dense, Flatten, MaxPooling1D, MaxPooling3D, Concatenate
import keras.backend as K
import numpy as np
import pickle
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import matplotlib.pyplot as plt
from core import util
from datetime import datetime
import os
import sys
newPath = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))) + os.path.sep + 'trading-gym'
sys.path.append(newPath)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = 0

"""
build q newtork using cnn and dense layer
"""


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def theil_u(y_true, y_pred):
    up = K.sqrt(K.mean(K.square(y_true - y_pred)))
    bottom = K.sqrt(K.mean(K.square(y_true))) + K.sqrt(K.mean(K.square(y_pred)))
    return up / bottom


def r(y_true, y_pred):
    mean_y_true = K.mean(y_true)
    mean_y_pred = K.mean(y_pred)

    up = K.sum((y_true - mean_y_true) * (y_pred - mean_y_pred))
    bottom = K.sqrt(K.sum(K.square(y_true - mean_y_true)) * K.sum(K.square(y_pred - mean_y_pred)))

    return up / bottom


def build_network(max_len=7, init_mode='uniform', neurons=30, activation='relu'):
    if activation == 'leaky_relu':
        input_order = Input(shape=(10, 2, 120, 2), name="x1")
        input_tranx = Input(shape=(120, 11), name="x2")
        input_left_time = Input(shape=(max_len,), name="x3")

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

    input_order = Input(shape=(10, 2, 120, 2), name="x1")
    input_tranx = Input(shape=(120, 11), name="x2")
    input_left_time = Input(shape=(max_len,), name="x3")

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


def get_real_data(date, ticker, max_len, save_dir, train_data_rows=None):

    x1_dimension_info = (10, 2, 120, 2)
    x2_dimension_info = (120, 11)
    x3_dimension_info = (max_len,)

    pickle_name = save_dir + os.path.sep + date + '_' + ticker + '.pickle'
    f = open(pickle_name, 'rb')
    d = pickle.load(f)  # d[data_type][second] : map object!!
    f.close()

    if train_data_rows is None:
        train_data_rows = len(d[0])

    x1 = np.zeros([10, 2, 120, 2])
    x2 = np.zeros([120, 11])
    x3 = np.zeros([max_len])

    d_x1 = []
    d_x2 = []
    d_x3 = []
    d_y1 = []

    for idx in range(train_data_rows):
        sys.stdout.write("\rloading data from ticker %s" %ticker + ", yyyymmdd %s" %date + "  %i" % idx + " / %i 완료" % train_data_rows)
        sys.stdout.flush()

        for second in range(x1_dimension_info[2]):  # 60: seconds
            tmp = d[0][idx][second]
            for row in range(x1_dimension_info[0]):  # 10 : row
                for column in range(x1_dimension_info[1]):  # 2 : column
                    for channel in range(x1_dimension_info[3]):  # 2 : channel
                        x1[row][column][second][channel] = tmp[channel * 20 + column * 10]
        d_x1.append(x1)

        for second in range(x2_dimension_info[0]):  # 120 : seconds
            tmp = d[1][idx][second]
            for feature in range(x2_dimension_info[1]):  # 11 : features
                x2[second, feature] = tmp[feature]
        d_x2.append(x2)

        binary_second = util.seconds_to_binary_array(d[2][idx], max_len)
        for feature in range(x3_dimension_info[0]):  # max_len :features
            x3[feature] = binary_second[feature]

        d_x3.append(x3)

        # for second in range(y1_dimension_info[0]): # 60 : seconds
        d_y1.append(d[3][idx])

    sys.stdout.write("\r")
    sys.stdout.flush()
    return np.asarray(d_x1), np.asarray(d_x2), np.asarray(d_x3), np.asarray(d_y1)


def train_using_real_data(directory, max_len, params, save_dir, train_dir):
    ## params
    batch_size = params['batch_size']
    epochs = params['epochs']
    neurons = params['neurons']
    activation = params["activation"]

    model = build_network(max_len, neurons=neurons, activation=activation)
    model.compile(optimizer='adam', loss='mse', metrics=[metrics.mae, metrics.mape, mean_pred, theil_u, r])
    model.summary()

    l = load_ticker_yyyymmdd_list_from_directory(directory)

    t_x1, t_x2, t_x3, t_y1 = [], [], [], []

    for (da, ti) in l:
        x1, x2, x3, y1 = get_real_data(da, ti, max_len, save_dir=save_dir)
        t_x1.append(x1)
        t_x2.append(x2)
        t_x3.append(x3)
        t_y1.append(y1)
        print('loading data from ticker {}, yyyymmdd {} is finished.'.format(ti, da))
    t_x1 = np.concatenate(t_x1)
    t_x2 = np.concatenate(t_x2)
    t_x3 = np.concatenate(t_x3)
    t_y1 = np.concatenate(t_y1)
    print('total x1 : {}, total x2 : {}, total x3 : {}, total y1 : {}'.format(len(t_x1), len(t_x2), len(t_x3), len(t_y1)))

    # {steps} --> this file will be saved whenver it runs every steps as much as {step}
    checkpoint_weights_filename = 'boa_{info}.{extension}'

    # model.load_weights(filepath = checkpoint_weights_filename.format(step='end'), by_name=True, skip_mismatch=True)

    # TODO: here we can add hyperparameters information like below!!
    log_filename = 'boa_{}_log.json'.format('fill_params_information_in_here')
    checkpoint_interval = 50

    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=checkpoint_interval)]
    callbacks += [FileLogger(log_filename, interval=100)]

    print('start to train.')
    history = model.fit({'x1': t_x1, 'x2': t_x2, 'x3': t_x3}, t_y1,
                        epochs=epochs, verbose=2, batch_size=batch_size, validation_split=0.1,
                        callbacks=callbacks)

    info = str(epochs) + 'epochs_' + str(batch_size) + 'batch_' + str(neurons) + 'neurons_' + 'act(' + str(
        activation) + ')'

    history_dir = train_dir + os.path.sep + 'history'
    if not os.path.isdir(history_dir):
        os.makedirs(history_dir)
    with open(history_dir + os.path.sep +
              checkpoint_weights_filename.format(info=info, extension='history'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    model.save(filepath=train_dir + os.path.sep + checkpoint_weights_filename.format(info=info, extension='h5'))
    model.save_weights(filepath=train_dir + os.path.sep + checkpoint_weights_filename.format(info=info, extension='h5f'))

    return history


def plot_history(history, to_plot, params, train_dir):

    plots_dir = train_dir + os.path.sep + 'plots'
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    batch_size = params['batch_size']
    epochs = params['epochs']
    neurons = params['neurons']
    activation = params["activation"]

    for key in to_plot.keys():

        file_name = 'boa_' + str(epochs) + 'epochs_' + str(batch_size) + 'batch_' + str(neurons) + 'neurons_' + \
                    'act(' + str(activation) + ')_' + key + '.png'
        category = to_plot[key]
        plt.plot(history[category])
        if key != 'Corr':
            plt.plot(history['val_' + category])
        plt.title(key)
        plt.ylabel(key)
        plt.xlabel('Epoch')
        if key != 'Corr':
            plt.legend(['Train', 'Validation'], loc='upper left')
        else:
            plt.legend(['Train'], loc='upper left')
        plt.savefig(plots_dir + os.path.sep + file_name)
        plt.gcf().clear()

def main(useHistory = False):

    # train_using_fake_data()
    # pickle path
    pickle_dir = 'pickles'
    directory = os.path.abspath(make_dir(os.path.dirname(os.path.abspath(__file__)), pickle_dir))

    train_dir = 'train'
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)

    # max length of bit for 120
    max_len = util.get_maxlen_of_binary_array(120)

    dict_to_plot = {
        'Loss': 'loss',
        'MAE': 'mean_absolute_error',
        'MAPE': 'mean_absolute_percentage_error',
        'Mean Pred': 'mean_pred',
        'Corr': 'r',
        "Theil's U": 'theil_u'
    }

    param_list = [
        # BSA best result
        # {'epochs': 50, 'batch_size': 64, 'neurons': 100, 'activation': 'leaky_relu'},
        # BOA best result
        {'epochs': 100, 'batch_size': 10, 'neurons': 15, 'activation': 'leaky_relu'},
        {'epochs': 10, 'batch_size': 10, 'neurons': 15, 'activation': 'leaky_relu'},
        {'epochs': 10, 'batch_size': 10, 'neurons': 20, 'activation': 'leaky_relu'},
        {'epochs': 10, 'batch_size': 10, 'neurons': 25, 'activation': 'leaky_relu'},
        {'epochs': 10, 'batch_size': 10, 'neurons': 30, 'activation': 'leaky_relu'}
        # SSA best result
        # {'activation': 'leaky_relu', 'batch_size': 70, 'epochs': 50, 'neurons': 125},
        # {'activation': 'leaky_relu', 'batch_size': 70, 'epochs': 75, 'neurons': 75},
        # {'activation': 'leaky_relu', 'batch_size': 70, 'epochs': 100, 'neurons': 125},
        # {'activation': 'leaky_relu', 'batch_size': 50, 'epochs': 100, 'neurons': 175},
        # {'activation': 'leaky_relu', 'batch_size': 30, 'epochs': 100, 'neurons': 175},
        # SOA best result
        # {'batch_size': 30, 'epochs': 75, 'max_len': 7, 'neurons': 80}
    ]

    for params in param_list:
        if useHistory:
            info = str(params['epochs']) + 'epochs_' + str(params['batch_size']) + 'batch_' + str(params['neurons']) + 'neurons_' + 'act(' + str(params['activation']) + ')'
            history_file = train_dir + os.path.sep + 'history' + os.path.sep + 'boa_{info}.{extension}'.format(info=info, extension='history')
            f = open(history_file, 'rb')
            history = pickle.load(f)
        else:
            history = train_using_real_data(directory, max_len, params, pickle_dir, train_dir)
            history = history.history
        plot_history(history, dict_to_plot, params, train_dir)


main(False)
