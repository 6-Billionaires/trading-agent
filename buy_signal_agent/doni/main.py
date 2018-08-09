from gym_core import tgym
from gym_core import ioutil
from collections import deque
import pandas as pd
import datetime
import numpy as np
import pickle

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

    pickle_name = current_date + '_' + current_ticker + '.pickle'
    f = open(pickle_name, 'wb')
    pickle.dump([x_2d, x_1d, y_1d], f)
    f.close()

prepare_datasets()