from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv1D, Dense, Flatten, MaxPool1D, MaxPool2D
import numpy as np

"""
build q newtork using cnn and dense layer
"""
def build_network():

    input_price = Input(shape=(60, 1, ))
    input_order_sell = Input(shape=(10, 2, 60,))
    input_order_buy  = Input(shape=(10, 2, 60,))
    input_tranx = Input(shape=(60, 11, ))

    h_conv1d_1 = Conv1D(filters=(1, 3), kernel_size=64, activation='relu')(input_price)
    h_conv1d_2 = Conv1D(filters=(1, 3), kernel_size=64, activation='relu')(input_tranx)

    h_conv1d_3 = MaxPool1D(h_conv1d_1)
    h_conv1d_4 = MaxPool1D(h_conv1d_2)

    h_conv1d_5 = Conv1D(filters=(1, 3), kernel_size=64, activation='relu')(h_conv1d_3)
    h_conv1d_6 = Conv1D(filters=(1, 3), kernel_size=64, activation='relu')(h_conv1d_4)

    h_conv1d_7 = MaxPool1D(h_conv1d_5)
    h_conv1d_8 = MaxPool1D(h_conv1d_6)

    o_conv1d_1 = Flatten(h_conv1d_7, h_conv1d_8)

    h_conv2d_1_1 = Conv2D(filters=(2, 1), kernel_size=64, activation='relu')(input_order_buy)
    h_conv2d_1_2 = Conv2D(filters=(1, 2), kernel_size=64, activation='relu')(input_order_buy)

    h_conv2d_1_3 = MaxPool1D(h_conv2d_1_1)
    h_conv2d_1_4 = MaxPool1D(h_conv2d_1_2)

    h_conv2d_1_5 = Conv2D(filters=(2, 1), kernel_size=64, activation='relu')(h_conv2d_1_3)
    h_conv2d_1_6 = Conv2D(filters=(1, 2), kernel_size=64, activation='relu')(h_conv2d_1_4)

    h_conv2d_1_7 = MaxPool1D(h_conv2d_1_5)
    h_conv2d_1_8 = MaxPool1D(h_conv2d_1_6)

    o_conv2d_1 = Flatten(h_conv2d_1_7, h_conv2d_1_8)

    h_conv2d_2_1 = Conv2D(filters=(2, 1), kernel_size=64, activation='relu')(input_order_sell)
    h_conv2d_2_2 = Conv2D(filters=(1, 2), kernel_size=64, activation='relu')(input_order_sell)

    h_conv2d_2_3 = MaxPool1D(h_conv2d_2_1)
    h_conv2d_2_4 = MaxPool1D(h_conv2d_2_2)

    h_conv2d_2_5 = Conv2D(filters=(2, 1), kernel_size=64, activation='relu')(h_conv2d_2_3)
    h_conv2d_2_6 = Conv2D(filters=(1, 2), kernel_size=64, activation='relu')(h_conv2d_2_4)

    h_conv2d_2_7 = MaxPool1D(h_conv2d_2_5)
    h_conv2d_2_8 = MaxPool1D(h_conv2d_2_6)

    o_conv2d_2 = Flatten(h_conv2d_2_7, h_conv2d_2_8)

    i_concatenated_all_h = Flatten(o_conv1d_1,o_conv2d_1, o_conv2d_2)

    output = Dense(1,activation='linear')(i_concatenated_all_h)

    model = Model([input_price, input_order_sell, input_order_buy, input_tranx], output)

    return model


def get_sample_data():
    np.arange(100).reshape([2,50])
