from gym_core import tgym
from gym_core import ioutil
from collections import deque
import config
import pandas as pd
import datetime
import numpy as np
import pickle
import random


def prepare_datasets(max_secs=120):
    l = ioutil.load_data_from_dicrectory('0')

    for li in l:
        prepare_dataset(li, max_secs)


def prepare_dataset(d, max_secs):
    current_date = d['meta']['date']
    current_ticker = d['meta']['ticker']

    c_start = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]), int(current_date[6:8]), 9, 5)
    c_end = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]), int(current_date[6:8]), 15, 20)
    c_rng_timestamp = pd.date_range(start=c_start, end=c_end, freq='S')

    x_1d_elapsed_secs = []
    x_2d = []
    x_1d = []
    y_1d = []

    first_quote = d['quote'].loc[c_rng_timestamp[0]]
    first_order = d['order'].loc[c_rng_timestamp[0]]

    for i, s in enumerate(c_rng_timestamp):
        if i % 3600 == 0:
            print(current_date, current_ticker, s)

        # SSA 에서 시그널을 받고 실제 주문을 하는데 까지 걸린 시간을 랜덤 생성.
        elapsed_secs = random.randint(0, max_secs)

        try:
            first_quote = d['quote'].loc[s]
            first_order = d['order'].loc[s]
        except KeyError as e:
            print('찾을 수 없는 key 값이 있습니다.', current_ticker, e)

        # calculate Y
        price_at_signal = d['quote'].loc[c_rng_timestamp[i-elapsed_secs]]['Price(last excuted)']
        price = d['quote'].loc[c_rng_timestamp[i]]['Price(last excuted)']

        x_1d_elapsed_secs.append(elapsed_secs)
        x_2d.append(first_order)
        x_1d.append(first_quote)
        y_1d.append(price - price_at_signal)

    pickle_name = current_date + '_' + current_ticker + '.pickle'
    f = open('./pickles/'+pickle_name, 'wb')
    pickle.dump([x_2d, x_1d, x_1d_elapsed_secs, y_1d], f)
    f.close()


prepare_datasets()