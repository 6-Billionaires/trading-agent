
import os
import sys
newPath = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))) + '/trading-gym'
sys.path.append(newPath)

from keras.models import Model
from keras.layers import LeakyReLU, Input, Dense, Conv3D, Conv1D, Dense, Flatten, MaxPooling1D, MaxPooling2D,MaxPooling3D,Concatenate


from gym_core.ioutil import *  # file i/o to load stock csv files

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.SSA_PARAMS['P_TRAINING_GPU'])
_len_observation = int(config.SSA_PARAMS['P_OBSERVATION_LEN'])
_pickle_evaluate_dir = config.SSA_PARAMS['PICKLE_DIR_FOR_TEST']

"""
it will prevent process not to occupying 100% of gpu memory for the first time. 
Instead, it will use memory incrementally.
"""

"""
build q newtork using cnn and dense layer
"""
def build_network_for_sparsed():

    input_order = Input(shape=(10, 2, _len_observation, 2), name="x1")
    input_tranx = Input(shape=(_len_observation, 11), name="x2")
    input_elapedtime = Input(shape=(max_len,), name="x3")
    input_lefttime = Input(shape=(max_len,), name="x4")

    h_conv1d_2 = Conv1D(filters=16, kernel_size=3)(input_tranx)
    h_conv1d_2 = LeakyReLU(alpha=0.3)(h_conv1d_2)

    h_conv1d_4 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(h_conv1d_2)

    h_conv1d_6 = Conv1D(filters=32, kernel_size=3)(h_conv1d_4)
    h_conv1d_6 = LeakyReLU(alpha=0.3)(h_conv1d_6)

    h_conv1d_8 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(h_conv1d_6)

    h_conv3d_1_1 = Conv3D(filters=16, kernel_size=(2, 1, 5))(input_order)
    h_conv3d_1_1 = LeakyReLU(alpha=0.3)(h_conv3d_1_1)

    h_conv3d_1_2 = Conv3D(filters=16, kernel_size=(1, 2, 5))(input_order)
    h_conv3d_1_2 = LeakyReLU(alpha=0.3)(h_conv3d_1_2)

    h_conv3d_1_3 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_1)
    h_conv3d_1_4 = MaxPooling3D(pool_size=(1, 1, 3))(h_conv3d_1_2)

    h_conv3d_1_5 = Conv3D(filters=32, kernel_size=(1, 2, 5))(h_conv3d_1_3)
    h_conv3d_1_5 = LeakyReLU(alpha=0.3)(h_conv3d_1_5)

    h_conv3d_1_6 = Conv3D(filters=32, kernel_size=(2, 1, 5))(h_conv3d_1_4)
    h_conv3d_1_6 = LeakyReLU(alpha=0.3)(h_conv3d_1_6)

    h_conv3d_1_7 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_5)
    h_conv3d_1_8 = MaxPooling3D(pool_size=(1, 1, 5))(h_conv3d_1_6)
    o_conv3d_1 = Concatenate(axis=-1)([h_conv3d_1_7, h_conv3d_1_8])

    o_conv3d_1_1 = Flatten()(o_conv3d_1)

    i_concatenated_all_h_1 = Flatten()(h_conv1d_8)

    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1, o_conv3d_1_1, input_elapedtime, input_lefttime])

    hidden_out = Dense(100)(i_concatenated_all_h)
    hidden_out = LeakyReLU(alpha=0.3)(hidden_out)

    output = Dense(1, activation='linear')(hidden_out)

    model = Model([input_order, input_tranx, input_elapedtime, input_lefttime], output)

    return model


def get_real_data_sparsed(pickle_dir, ticker='001470', date='20180420', train_data_rows=None):
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

# train_using_real_data(d, 'sparse')
max_len = get_maxlen_of_binary_array(120)

model = build_network_for_sparsed()
model.compile(optimizer='adam', loss='mse', metrics=['mae','mape','accuracy'])
model.summary()
model.load_weights('final_weight.h5f')

# model = Model('final_model.h5')

l = load_ticker_yyyymmdd_list_from_directory(_pickle_evaluate_dir)
for (ti, da) in l:
    x1, x2, x3, x4, y = get_real_data_sparsed(pickle_dir=_pickle_evaluate_dir, ticker=ti, date=da)
    scores = model.evaluate({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4}, y, verbose=0)
    print("%s: %.2f    %s: %.2f    %s: %.2f" % (model.metrics_names[1], scores[1], model.metrics_names[2], scores[2], model.metrics_names[3], scores[3]))
