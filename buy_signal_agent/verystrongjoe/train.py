
from keras.models import Model
from keras.layers import LeakyReLU, Input, Dense, Conv3D, Conv1D, Dense, Flatten, MaxPooling1D, MaxPooling2D,MaxPooling3D,Concatenate
import numpy as np
import pickle
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from gym_core.ioutil import *  # file i/o to load stock csv files
import logging
from core.scikit_learn_multi_input import KerasRegressor
from sklearn.model_selection import GridSearchCV



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
def build_network_for_sparsed(optimizer='adam',init_mode='uniform', filters=16, neurons=20, activation='relu'):
    if activation == 'leaky_relu':
        activation = LeakyReLU(alpha=0.3)

    input_order = Input(shape=(10, 2, 60, 2), name="x1")
    input_tranx = Input(shape=(60, 11), name="x2")

    h_conv1d_2 = Conv1D(filters=16, kernel_initializer=init_mode, kernel_size=3, activation=activation)(input_tranx)
    h_conv1d_4 = MaxPooling1D(pool_size=3,  strides=None, padding='valid')(h_conv1d_2)
    h_conv1d_6 = Conv1D(filters=32, kernel_initializer=init_mode, kernel_size=3, activation=activation)(h_conv1d_4)
    h_conv1d_8 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(h_conv1d_6)

    h_conv3d_1_1 = Conv3D(filters=filters, kernel_initializer=init_mode, kernel_size=(2, 1, 5), activation=activation)(input_order)
    h_conv3d_1_2 = Conv3D(filters=filters,  kernel_initializer=init_mode,kernel_size=(1, 2, 5), activation=activation)(input_order)

    h_conv3d_1_3 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_1)
    h_conv3d_1_4 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_2)

    h_conv3d_1_5 = Conv3D(kernel_initializer=init_mode, filters=filters*2, kernel_size=(1, 2, 5), activation=activation)(h_conv3d_1_3)
    h_conv3d_1_6 = Conv3D(kernel_initializer=init_mode, filters=filters*2, kernel_size=(2, 1, 5), activation=activation)(h_conv3d_1_4)

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




def get_real_data_sparsed(ticker='001470', date='20180420', train_data_rows=None, save_dir=''):
    """
    Get sparsed data for supervised learning

    :param ticker: ticker number to read
    :param date: date yyyymmdd to read
    :param train_data_rows: data rows to read for training, default None : read all rows
    :param save_dir: default ''
    :return: training data x1, x2, y1 for supervised learning model
    """
    current_ticker = ticker
    current_date = date

    x1_dimension_info = (10, 2, 60, 2)  # 60 --> 120 (@iljoo)
    x2_dimension_info = (60, 11)
    # y1_dimension_info = (120,)

    pickle_name = save_dir + os.path.sep + current_ticker + '_' + current_date + '.pickle'
    f = open(pickle_name, 'rb')
    d = pickle.load(f)  # d[data_type][second] : mapobject!!
    f.close()

    total_rows = len(d[0])

    if train_data_rows is None:
        train_data_rows = len(d[0])

    x1 = np.zeros([10, 2, 60, 2])
    x2 = np.zeros([60, 11])
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


def get_real_data(ticker='001470', date='20180420', train_data_rows=None, save_dir=''):
    """
    Get data for supervised learning

    :param ticker: ticker number to read
    :param date: date yyyymmdd to read
    :param train_data_rows: data rows to read for training, default None : read all rows
    :param save_dir: default ''
    :return: training data x1, x2, y1 for supervised learning model
    """
    current_ticker = ticker
    current_date = date

    x1_dimension_info = (10, 2, 120, 2)  # 60 --> 120 (@iljoo)
    x2_dimension_info = (120, 11)
    # y1_dimension_info = (120,)

    pickle_name = save_dir + os.path.sep + current_ticker + '_' + current_date + '.pickle'
    f = open(pickle_name, 'rb')
    d = pickle.load(f)  # d[data_type][second] : mapobject!!
    f.close()

    total_rows = len(d[0])

    if train_data_rows is None:
        train_data_rows = len(d[0])

    x1 = np.zeros([10, 2, 120, 2])
    x2 = np.zeros([120, 11])
    y1 = np.zeros([120])

    d_x1 = []
    d_x2 = []
    d_y1 = []

    for idx_second in range(train_data_rows):

        if idx_second + 120 > total_rows:
            break

        for row in range(x1_dimension_info[0]):  # 10 : row
            for column in range(x1_dimension_info[1]):  # 2 : column
                for second in range(x1_dimension_info[2]):  # 120 : seconds
                    for channel in range(x1_dimension_info[3]):  # 2 : channel
                        key = ''
                        if channel == 1:
                            key = 'Buy'
                        else:
                            key = 'Sell'
                        if column == 0:
                            value = 'Hoga'
                        else:
                            value = 'Order'

                        x1[row][column][second][channel] = d[0][idx_second+second][key+value+str(row+1)]
        d_x1.append(x1)

        for second in range(x2_dimension_info[0]):  # 120 : seconds
            for feature in range(x2_dimension_info[1]):  # 11 : features
                x2[second, feature] = d[1][idx_second+second][feature]
        d_x2.append(x2)

        # for second in range(y1_dimension_info[0]): # 60 : seconds
        d_y1.append(d[2][idx_second])

    return np.asarray(d_x1), np.asarray(d_x2), np.asarray(d_y1)

# def train_using_fake_data():
#     train_per_each_episode('','',True)


def train_using_real_data(d, save_dir=''):

    model = build_network()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()

    l = load_ticker_yyyymmdd_list_from_directory(d)

    t_x1, t_x2, t_y1 = [], [], []

    for (ti, da) in l:
        print('loading data from ticker {}, yyyymmdd {} is started.'.format(ti, da))
        x1, x2, y1 = load_data(ti, da, use_fake_data=False, save_dir=save_dir)
        t_x1.append(x1)
        t_x2.append(x2)
        t_y1.append(y1)
        print('loading data from ticker {}, yyyymmdd {} is finished.'.format(ti, da))

    print('total x1 : {}, total x2 : {}, total y1 : {}'.format(len(t_x1), len(t_x2), len(t_y1)))

    # {steps} --> this file will be saved whenever it runs every steps as much as {step}
    checkpoint_weights_filename = 'bsa_' + 'fill_params_information_in_here' + '_weights_{step}.h5f'

    # TODO: here we can add hyperparameters information like below!!
    log_filename = 'bsa_{}_log.json'.format('fill_params_information_in_here')
    checkpoint_interval = 50

    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=checkpoint_interval)]
    callbacks += [FileLogger(log_filename, interval=100)]

    print('start to train.')
    model.fit({'x1': t_x1, 'x2': t_x2}, t_y1, epochs=50, verbose=2, batch_size=64, callbacks=callbacks)
    model.save_weights('final_weight.h5f')

def train_using_real_data_sparsed(d, save_dir=''):

    # model = build_network_for_sparsed()

    l = load_ticker_yyyymmdd_list_from_directory(d)

    t_x1, t_x2, t_y1 = [],[],[]

    for (ti, da) in l:
        print('loading data from ticker {}, yyyymmdd {} is started.'.format(ti, da))
        x1, x2, y1 = load_data_sparsed(ti, da, use_fake_data=False, save_dir=save_dir)
        t_x1.append(x1)
        t_x2.append(x2)
        t_y1.append(y1)
        print('loading data from ticker {}, yyyymmdd {} is finished.'.format(ti, da))
    t_x1 = np.concatenate(t_x1)
    t_x2 = np.concatenate(t_x2)
    t_y1 = np.concatenate(t_y1)

    print('total x1 : {}, total x2 : {}, total y1 : {}'.format(len(t_x1), len(t_x2), len(t_y1)))
    # np.append(t_x1,values=(36,8,10,2,60))

    # {steps} --> this file will be saved whenever it runs every steps as much as {step}
    checkpoint_weights_filename = 'bsa_' + 'fill_params_information_in_here' + '_weights_{step}.h5f'

    # TODO: here we can add hyperparameters information like below!!
    log_filename = 'bsa_{}_log.json'.format('fill_params_information_in_here')
    checkpoint_interval = 50

    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=checkpoint_interval)]
    callbacks += [FileLogger(log_filename, interval=100)]

    print('start to train.')
    # model.fit({'x1': t_x1, 'x2': t_x2}, t_y1, epochs=50, verbose=2, batch_size=64, callbacks=callbacks)
    # model.save_weights('final_weight.h5f')

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

    # todo : second try!
    # batch_size = [10]
    # epochs = [70, 100]
    # neurons = [70, 100]
    # activation = ['leaky_relu']

    # todo : third try!
    batch_size = [10]
    epochs = [70]
    neurons = [100, 120, 150]
    activation = ['leaky_relu']



    # first try
    # end up with Best: -46695.504027 using {'activation': 'leaky_relu', 'batch_size': 10, 'epochs': 50, 'neurons': 50}
    # batch_size = [10, 20, 40, 100]
    # epochs = [10, 50]
    # neurons = [20, 25, 50]
    # activation = ['relu', 'leaky_relu', 'tanh']






    param_grid = dict(batch_size=batch_size, epochs=epochs, neurons=neurons, activation=activation)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    # grid_result = grid.fit({'x1': t_x1, 'x2': t_x2}, t_y1)

    grid_result = grid.fit(
        np.array(  [ {'x1': a, 'x2': b}  for a, b in zip(t_x1, t_x2)]),
        t_y1)

    # grid_result = grid.fit(
    #     np.array([{'x1': a, 'x2': b}] for a, b in zip(t_x1, t_x2)),
    #     t_y1)


    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))



@runtime
def load_data(t, d, use_fake_data=False, save_dir =''):
    if use_fake_data:
        x1, x2, y = get_sample_data(10)
    else:
        current_date = d
        current_ticker = t
        #if you give second as None, it will read every seconds in file.
        # x1, x2, y = get_real_data(current_ticker, current_date, train_data_rows=130)
        x1, x2, y = get_real_data(current_ticker, current_date, save_dir=save_dir)
    return x1, x2, y


@runtime
def load_data_sparsed(t, d, use_fake_data=False, save_dir =''):
    if use_fake_data:
        x1, x2, y = get_sample_data(10)
    else:
        current_date = d
        current_ticker = t
        #if you give second as None, it will read every seconds in file.
        # x1, x2, y = get_real_data(current_ticker, current_date, train_data_rows=130)
        x1, x2, y = get_real_data_sparsed(current_ticker, current_date, save_dir=save_dir)
    return x1, x2, y

# train_using_fake_data()
# d  = 'D:\\dev\\workspace\\trading-agent\\buy_signal_agent\\verystrongjoe\\sparse_2'
d = 'C:\\Git\\trading-agent\\buy_signal_agent\\verystrongjoe\\sparse_2'
# train_using_real_data(d, 'sparse')
train_using_real_data_sparsed(d, 'sparse_2')



# if __name__ == '__main__' :
#     batch_size = [10, 20, 40, 60, 80, 100]
#     epochs = [10, 50, 100]
