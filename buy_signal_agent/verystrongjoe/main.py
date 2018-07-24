from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv1D, Dense, Flatten, MaxPool1D, MaxPool2D
from gym_core import tgym
from gym_core import ioutil
from collections import deque
import config
import pandas as pd
import datetime
import numpy as np
import pickle

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

def prepare_datasets(secs=60):
    l = ioutil.load_data_from_dicrectory('0')
    for li in l:
        prepare_dataset(li, secs)

def prepare_dataset(d, secs):
    current_date    = d['meta']['date']
    current_ticker  = d['meta']['ticker']

    d_price = deque(maxlen=secs)

    c_start = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]), int(current_date[6:8]), 9, 5)
    c_end = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]), int(current_date[6:8]), 15, 20)
    c_rng_timestamp = pd.date_range(start=c_start, end=c_end, freq='S')

    x_2d = []
    x_1d = []
    y_1d = []

    for i, s in enumerate(c_rng_timestamp):

        end = i+secs;

        if len(c_rng_timestamp) < end:
            break
        else:
            first_quote = d['quote'].loc[s]
            first_order = d['order'].loc[s]

            j = i
            width = 0
            # calculate Y
            for j in range(secs):
                if j == 0:
                    price_at_signal = d['quote'].loc[c_rng_timestamp[j]]['Price(last excuted)']
                else:
                    price = d['quote'].loc[c_rng_timestamp[j]]['Price(last excuted)']
                    gap = price - price_at_signal
                    width += gap
            x_2d.append(first_order)
            x_1d.append(first_quote)
            y_1d.append(width)
    # return
    pickle_name = current_date + '_' + current_ticker + '.pickle'
    f = open(pickle_name, 'wb')
    pickle.dump([x_2d, x_1d, y_1d], f)
    f.close()

prepare_datasets()