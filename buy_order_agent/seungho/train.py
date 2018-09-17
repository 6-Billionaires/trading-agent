import os, sys
from keras.models import Model
from keras.layers import Input, Conv3D, Conv1D, Dense, Flatten, MaxPooling1D, MaxPooling3D, Concatenate
import numpy as np
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from gym_core import ioutil  # file i/o to load stock csv files
import random
from core import util
# from core.scikit_learn_multi_input_boa import KerasRegressor
# from sklearn.model_selection import GridSearchCV

"""
build q newtork using cnn and dense layer
"""


def build_network(optimizer='adam', init_mode='uniform', filters=16, neurons=20, max_secs = 90, max_len = 7):
    input_order = Input(shape=(10, 2, max_secs, 2), name="x1")
    input_tranx = Input(shape=(max_secs, 11), name="x2")
    input_remain_secs = Input(shape=(max_len, ), name="x3")

    h_conv1d_2 = Conv1D(filters=16, kernel_initializer=init_mode, kernel_size=3, activation='relu')(input_tranx)
    h_conv1d_4 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(h_conv1d_2)
    h_conv1d_6 = Conv1D(filters=32, kernel_initializer=init_mode, kernel_size=3, activation='relu')(h_conv1d_4)
    h_conv1d_8 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(h_conv1d_6)

    h_conv3d_1_1 = Conv3D(filters=filters, kernel_initializer=init_mode, kernel_size=(2, 1, 5), activation='relu')(
        input_order)
    h_conv3d_1_2 = Conv3D(filters=filters, kernel_initializer=init_mode, kernel_size=(1, 2, 5), activation='relu')(
        input_order)

    h_conv3d_1_3 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_1)
    h_conv3d_1_4 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_2)

    h_conv3d_1_5 = Conv3D(kernel_initializer=init_mode, filters=filters * 2, kernel_size=(1, 2, 5), activation='relu')(
        h_conv3d_1_3)
    h_conv3d_1_6 = Conv3D(kernel_initializer=init_mode, filters=filters * 2, kernel_size=(2, 1, 5), activation='relu')(
        h_conv3d_1_4)

    h_conv3d_1_7 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_5)
    h_conv3d_1_8 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_6)
    o_conv3d_1 = Concatenate(axis=-1)([h_conv3d_1_7, h_conv3d_1_8])

    o_conv3d_1_1 = Flatten()(o_conv3d_1)

    i_concatenated_all_h_1 = Flatten()(h_conv1d_8)

    # i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1])
    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1, input_remain_secs])

    i_concatenated_all_h = Dense(neurons, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

    output = Dense(1, kernel_initializer=init_mode, activation='linear')(i_concatenated_all_h)

    model = Model([input_order, input_tranx, input_remain_secs], output)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary()

    return model


def get_real_data(max_secs, max_len, csv, pickles, max_stock = 5):
    x1_dimension_info = (10, 2, max_secs, 2)
    x2_dimension_info = (max_secs, 11)
    x3_dimension_info = (max_len,)

    d_x1 = []
    d_x2 = []
    d_x3 = []
    d_y1 = []

    keys = list(pickles.keys())

    random.shuffle(keys)

    max_stock = max_stock if max_stock < len(keys) else len(keys)
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


def train_using_real_data(data, max_secs, max_len, isGrid=False):
    # {일자 + 종목코드} : {meta, quote, order}
    csv = ioutil.load_data_from_directory2('0')
    # {일자 + 종목코드} : [second, left_time, elapsed_time, y]
    pickles, max_size = ioutil.load_ticker_yyyymmdd_list_from_directory2(data)


    # {steps} --> this file will be saved whenver it runs every steps as much as {step}
    checkpoint_weights_filename = 'boa_weights_{step}.h5'

    #model.load_weights(filepath = checkpoint_weights_filename.format(step=0), by_name=True, skip_mismatch=True)

    # TODO: here we can add hyperparameters information like below!!
    log_filename = 'boa_{}_log.json'.format(max_secs)
    checkpoint_interval = 50

    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=checkpoint_interval)]
    callbacks += [FileLogger(log_filename, interval=100)]

    x1, x2, x3, y = get_real_data(max_secs, max_len, csv, pickles, 28)

    if isGrid == False:
        # def build_network(optimizer='adam', init_mode='uniform', filters=16, neurons=20, max_secs=90, max_len=7):
        model = build_network('adam', 'uniform', 16, 20, max_secs, max_len)
        # print('shape : ', x1.shape, x2.shape, y.shape)
        model.fit({'x1': x1, 'x2': x2, 'x3': x3}, y, epochs=60, verbose=2, batch_size=10, callbacks=callbacks)
        model.save('final_weight.h5')
    # else:
    #     print('start to train.')
    #     # model.fit({'x1': t_x1, 'x2': t_x2}, t_y1, epochs=50, verbose=2, batch_size=64, callbacks=callbacks)
    #     # model.save_weights('final_weight.h5f')
    #
    #     # create model
    #     model = KerasRegressor(build_fn=build_network, verbose=0)
    #     # define the grid search parameters
    #     batch_size = [10, 20, 40, 60, 80, 100]
    #     epochs = [10, 50, 100]
    #     neurons = [15, 20, 25, 30]
    #     param_grid = dict(batch_size=batch_size, epochs=epochs, neurons=neurons)
    #
    #     grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    #     # grid_result = grid.fit({'x1': t_x1, 'x2': t_x2}, t_y1)
    #
    #     # grid_result = grid.fit(np.array([{'x1': a, 'x2': b} for a, b in zip(x1, x2)]), y)
    #     grid_result = grid.fit(np.array([{'x1': a, 'x2': b, 'x3': c} for a, b, c in zip(x1, x2, x3)]), y)
    #
    #     # summarize results
    #     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #     means = grid_result.cv_results_['mean_test_score']
    #     stds = grid_result.cv_results_['std_test_score']
    #     params = grid_result.cv_results_['params']
    #     for mean, stdev, param in zip(means, stds, params):
    #         print("%f (%f) with: %r" % (mean, stdev, param))




def main():
    # train_using_fake_data()
    # picke path
    directory = os.path.abspath(ioutil.make_dir(os.path.dirname(os.path.abspath(__file__)), 'pickles'))
    # max length of bit for 90
    max_secs = 90
    max_len = util.get_maxlen_of_binary_array(max_secs)
    train_using_real_data(directory, max_secs, max_len)


main()