
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
from gym_core.ioutil import *  # file i/o to load stock csv files
import sell_signal_agent.ssa_metrics as mt


if args.device is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.BSA_PARAMS['P_TRAINING_GPU'])
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
_len_observation = int(config.BSA_PARAMS['P_OBSERVATION_LEN'])
_pickle_evaluate_dir = config.BSA_PARAMS['PICKLE_DIR_FOR_TEST']

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
        'epochs' : 70,
        'batchsize' : 10,
        'neurons' : 100,
        'activation' : 'leaky_relu'
    },
    {
        'epochs' : 70,
        'batchsize' : 20,
        'neurons' : 100,
        'activation' : 'leaky_relu'
    },
    {
        'epochs' : 100,
        'batchsize' : 10,
        'neurons' : 70,
        'activation' : 'leaky_relu'
    },
    {
        'epochs' : 70,
        'batchsize' : 10,
        'neurons' : 150,
        'activation' : 'leaky_relu'
    },
    {
        'epochs' : 70,
        'batchsize' : 10,
        'neurons' : 120,
        'activation' : 'leaky_relu'
    }
]


param_epochs = model_params[model_index]["epochs"]
param_batchsize = model_params[model_index]["batchsize"]
param_neurons = model_params[model_index]["neurons"]

name_subfix = '_e' + str(param_epochs) + '_b' + str(param_batchsize) + '_n' + str(param_neurons)


"""
it will prevent process not to occupying 100% of gpu memory for the first time. 
Instead, it will use memory incrementally.
"""

"""
build q newtork using cnn and dense layer
"""
def build_network_for_sparsed(optimizer='adam',init_mode='uniform',
                              filters=16, neurons=100, activation='relu', bsa_model_params=model_params[0]):

    if activation == 'leaky_relu':
        activation = LeakyReLU(alpha=0.3)

    neurons = bsa_model_params['neurons']

    input_order = Input(shape=(10, 2, _len_observation, 2), name="x1")
    input_tranx = Input(shape=(_len_observation, 11), name="x2")

    h_conv1d_2 = Conv1D(filters=filters, kernel_initializer=init_mode, kernel_size=3)(input_tranx)
    h_conv1d_2 = LeakyReLU(alpha=0.3)(h_conv1d_2)

    h_conv1d_4 = MaxPooling1D(pool_size=3,  strides=None, padding='valid')(h_conv1d_2)

    h_conv1d_6 = Conv1D(filters=filters*2, kernel_initializer=init_mode, kernel_size=3)(h_conv1d_4)
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
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', 'mae', 'mape', mt.mean_pred, mt.theil_u, mt.r])
    model.summary()

    return model

def get_real_data_sparsed(pickle_dir, ticker='001470', date='20180420', train_data_rows=None):
    """
    Get sparsed data for supervised learning
    :param dir : directory where pickle files exist
    :param ticker: ticker number to read
    :param date: date yyyymmdd to read
    :param train_data_rows: data rows to read for training, default None : read all rows
    :return: training data x1, x2, y1 for supervised learning model
    """
    current_ticker = ticker
    current_date = date

    x1_dimension_info = (10, 2, _len_observation, 2)  # 60 --> 120 (@iljoo)
    x2_dimension_info = (_len_observation, 11)
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

model = build_network_for_sparsed(bsa_model_params=model_params[model_index])
model.load_weights('weight' + name_subfix + '.h5f')

l = load_ticker_yyyymmdd_list_from_directory(_pickle_evaluate_dir)
t_x1, t_x2, t_y1 = [],[],[]

for (ti, da) in l:
    print('loading data from ticker {}, yyyymmdd {} is started.'.format(ti, da))
    x1, x2, y1 = get_real_data_sparsed(pickle_dir=_pickle_evaluate_dir, ticker=ti, date=da)
    t_x1.append(x1)
    t_x2.append(x2)
    t_y1.append(y1)
    print('loading data from ticker {}, yyyymmdd {} is finished.'.format(ti, da))
t_x1 = np.concatenate(t_x1)
t_x2 = np.concatenate(t_x2)
t_y1 = np.concatenate(t_y1)


scores = model.evaluate({'x1': t_x1, 'x2': t_x2}, t_y1, verbose=00)
print("%s: %.2f    %s: %.2f    %s: %.2f    %s: %.2f" % (model.metrics_names[1], scores[1], model.metrics_names[2], scores[2], model.metrics_names[3], scores[3], model.metrics_names[4], scores[4]))

with open('ssa_evaluate_model_history'+name_subfix, 'wb') as file_pi:
    pickle.dump(scores, file_pi)


