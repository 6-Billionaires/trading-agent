from keras.models import Model
from keras.layers import Input, Dense, Conv3D, Conv1D, Dense, Flatten, MaxPooling1D, MaxPooling2D,MaxPooling3D,Concatenate
import numpy as np

"""
build q newtork using cnn and dense layer
"""
def build_network():

    input_tranx = Input(shape=(60, 11), name="x1")
    input_order = Input(shape=(2, 10, 2, 60), name="x2")

    h_conv1d_2 = Conv1D(filters=32, kernel_size=3, activation='relu')(input_tranx)
    h_conv1d_4 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(h_conv1d_2)
    h_conv1d_6 = Conv1D(filters=32, kernel_size=3, activation='relu')(h_conv1d_4)
    h_conv1d_8 = MaxPooling1D(pool_size=2, strides=None, padding='valid')(h_conv1d_6)

    h_conv3d_1_1 = Conv3D(filters=32, kernel_size=(2, 1, 5), activation='relu', data_format="channels_first")(input_order)
    h_conv3d_1_2 = Conv3D(filters=32, kernel_size=(1, 2, 5), activation='relu', data_format="channels_first")(input_order)

    h_conv3d_1_3 = MaxPooling3D(pool_size=(1, 2, 2), data_format="channels_first")(h_conv3d_1_1)
    h_conv3d_1_4 = MaxPooling3D(pool_size=(2, 1, 2), data_format="channels_first")(h_conv3d_1_2)

    h_conv3d_1_5 = Conv3D(filters=16, kernel_size=(1, 1, 5), activation='relu', data_format="channels_first")(h_conv3d_1_3)
    h_conv3d_1_6 = Conv3D(filters=16, kernel_size=(1, 1, 5), activation='relu', data_format="channels_first")(h_conv3d_1_4)

    h_conv3d_1_7 = MaxPooling3D(pool_size=(1, 1, 10), data_format="channels_first")(h_conv3d_1_5)
    h_conv3d_1_8 = MaxPooling3D(pool_size=(1, 1, 10), data_format="channels_first")(h_conv3d_1_6)

    o_conv3d_1_1 = Flatten()(h_conv3d_1_7)
    o_conv3d_1_2 = Flatten()(h_conv3d_1_8)
    o_conv3d_1 = Concatenate()([o_conv3d_1_1, o_conv3d_1_2])

    i_concatenated_all_h_1 = Flatten()(h_conv1d_8)
    # i_concatenated_all_h_2 = Flatten()(o_conv3d_1)

    i_concatenated_all_h = Concatenate()([i_concatenated_all_h_1,o_conv3d_1])

    output = Dense(1,activation='linear')(i_concatenated_all_h)

    model = Model([input_order, input_tranx], output)

    return model


def get_sample_data(count):
    start = 0
    ld_x1 = []
    ld_x2 = []
    ld_y = []
    d1 = []
    for i in range(count):
        # d1_1 = np.arange(start, start + 2 * 10 * 60).reshape([10, 2, 60])
        # start += 2 * 10 * 60
        d1_2 = np.arange(start, start + 2 * 10 * 60 * 2).reshape([2, 10, 2, 60])
        start += 2 * 10 * 60
        d2 = np.arange(start, start+11*60).reshape([60, 11])
        start += 11 * 60
        ld_x1.append(d1_2)

        # ld_x1.append([d1_1, d1_2])
        ld_x2.append(d2)

    for j in range(count):
        d1 = np.arange(start, start + 60).reshape([60,])
        ld_y.append(d1)

    return ld_x1, ld_x2, ld_y

def train():

    x1, x2,  y = get_sample_data(10)

    model = build_network()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # model.fit([x1, x2], y)
    # model.fit({'x1': x1, 'x2': x2}, y)

train()