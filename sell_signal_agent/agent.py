from keras.models import Model
from keras.layers import Input, Dense, Conv3D, Conv1D, Dense, Flatten, MaxPooling1D, MaxPooling2D,MaxPooling3D,Concatenate
import numpy as np
import pickle

"""
build q newtork using cnn and dense layer
"""
def build_network():

    input_tranx = Input(shape=(60, 11), name="x1")
    input_order = Input(shape=(10, 2, 60, 2), name="x2")

    h_conv1d_2 = Conv1D(filters=32, kernel_size=3, activation='relu')(input_tranx)
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

    o_conv3d_1_1 = Flatten()(o_conv3d_1 )
#    o_conv3d_1_2 = Flatten()(h_conv3d_1_8)
#    o_conv3d_1 = Concatenate()([o_conv3d_1_1, o_conv3d_1_2])

    i_concatenated_all_h_1 = Flatten()(h_conv1d_8)
    # i_concatenated_all_h_2 = Flatten()(o_conv3d_1)

    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1,o_conv3d_1_1])

    output = Dense(1,activation='linear')(i_concatenated_all_h)

    model = Model([input_order, input_tranx], output)

    return model


def get_real_data(date='20180503', ticker='000430', seconds=None):

    current_date = date
    current_ticker = ticker

    ## (60, 11)  = a[1]
    x1_dimension_info = (10, 2, 60, 2)
    x2_dimension_info = (60, 11)
    y1_dimension_info = (60,)

    pickle_name = current_date + '_' + current_ticker + '.pickle'
    f = open('pickles\\'+pickle_name, 'rb')
    d = pickle.load(f) # d[data_type][second] : mapobject!!
    f.close()

    if seconds is None:
        seconds = len(d[0])

    x1 = np.zeros([10,2,60,2])
    x2 = np.zeros([60,11])
    y1 = np.zeros([60])

    d_x1 = []
    d_x2 = []
    d_y1 = []

    for idx_second in range(seconds):
        for row in range(x1_dimension_info[0]): #10 : row
            for column in range(x1_dimension_info[1]): #2 : column
                for second in range(x1_dimension_info[2]): #60 : seconds
                    for channel in range(x1_dimension_info[3]): #2 : channel
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
        for second in range(x2_dimension_info[0]):  #60 : seconds
            for feature in range(x2_dimension_info[1]):  #11 :features
                x2[second, feature] = d[1][idx_second+second][feature]
        d_x2.append(x2)

        # for second in range(y1_dimension_info[0]): #60 : seconds
        d_y1.append(d[2][idx_second])

    return np.asarray(d_x1), np.asarray(d_x2), np.asarray(d_y1)


def train():
    # x1, x2, y = get_sample_data(10)

    current_date ='20180503' # date
    current_ticker = '000430' # ticker

    # x1, x2, y = get_real_data(current_date,current_ticker,100)
    #if you give second as None, it will read every seconds in file.
    x1, x2, y = get_real_data(current_date, current_ticker)
    model = build_network()
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    model.summary()
    # model.fit([x1, x2], y)
    model.fit({'x1': x2, 'x2': x1}, y)

train()