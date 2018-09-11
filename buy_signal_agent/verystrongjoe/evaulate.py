from keras.models import Model
from keras.layers import Input, Dense, Conv3D, Conv1D, Dense, Flatten, MaxPooling1D, MaxPooling2D,MaxPooling3D,Concatenate
import numpy as np
import pickle
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from gym_core import ioutil  # file i/o to load stock csv files

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
def build_network_for_sparsed():
    input_order = Input(shape=(10, 2, 60, 2), name="x1")
    input_tranx = Input(shape=(60, 11), name="x2")

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
        d1_2 = np.arange(start, start + 2 * 10 * 60 * 2).reshape([10, 2, 60, 2])
        start += 2 * 10 * 60
        d2 = np.arange(start, start+11*60).reshape([60, 11])
        start += 11 * 60
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

model = build_network_for_sparsed()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# load weight
model.load_weights('final_weight.h5f')

x1, x2, y = get_sample_sparsed_data(10) #for test

scores = model.evaluate({'x1': x1, 'x2': x2}, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
