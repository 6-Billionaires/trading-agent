"""
UPDATE : here I will write down comments in detail for others to code their owns.
"""

# from gym_core import tgym
from gym_core import ioutil  # file i/o to load stock csv files
from collections import deque
import config
import pandas as pd
import datetime
import numpy as np
import pickle

"""
previously,  I gave secs as 120. but like ilzoo said, it needs to be 120.

in other agents, it can changes but you don't have to read every time seconds periods changes
just read maximum periods of data and reuse it   
"""
def prepare_datasets(secs=120):
    """
    :param secs:
        max_n_episode : you can give maximum count to read file for quick test!!
    :return:
    """
    # l = ioutil.load_data_from_directory('0', max_n_episode=1) # episode type
    l = ioutil.load_data_from_directory('0')  # episode type
    for li in l:
        prepare_dataset(li, secs)

def prepare_dataset(d, secs):
    current_date    = d['meta']['date']
    current_ticker  = d['meta']['ticker']

    d_price = deque(maxlen=secs)

    c_start = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]), int(current_date[6:8]), 9, 5) # 9hr 5min 0sec, start time
    c_end = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]), int(current_date[6:8]), 15, 20) # 15hr 20min 0sec, finish time
    c_rng_timestamp = pd.date_range(start=c_start, end=c_end, freq='S') # range between c_start and c_end saving each seconds' data

    x_2d = []  # orderbook
    x_1d = []  # transactions
    y_1d = []  # widht

    for i, s in enumerate(c_rng_timestamp):
        end = i+secs;

        if len(c_rng_timestamp) < i+secs:
            break

        else:
            first_quote = d['quote'].loc[s]
            first_order = d['order'].loc[s]
            width = 0
            threshold = 0.33  # TODO : @terryjo check this value!!

            # calculate width
            for j in range(secs):
                if j == 0:
                    # price_at_signal is the price when the current stock received signal
                    price_at_signal = d['quote'].loc[c_rng_timestamp[i+j]]['Price(last excuted)']
                else:
                    price = d['quote'].loc[c_rng_timestamp[i+j]]['Price(last excuted)']
                    # TODO : we need to subtract threshold from width considering transaction cost and fee.
                    gap = price - price_at_signal - threshold
                    width += gap

            x_2d.append(first_order)
            x_1d.append(first_quote)
            y_1d.append(width)

    pickle_name = current_date + '_' + current_ticker + '.pickle'
    f = open(pickle_name, 'wb')
    pickle.dump([x_2d, x_1d, y_1d], f)
    f.close()

prepare_datasets()