from gym_core import ioutil
from collections import deque
import pandas as pd
import datetime
import numpy as np
import pickle
import os
import sys
newPath = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))) + os.path.sep + 'trading-gym'
sys.path.append(newPath)



def prepare_datasets(interval=120, len_sequence_secs=120, save_dir='pickles', max_secs=90, threshold=0.33):
    l = ioutil.load_data_from_directory('0')
    for li in l:
        prepare_dataset(li, interval, len_sequence_secs, save_dir, max_secs, threshold)


def prepare_dataset(d, interval, len_sequence_secs, save_dir, max_secs, threshold):
    """
    original version
    loading data from ticker 20180403, yyyymmdd 003350 is started.
    executed time :  1538.569629558868 -> 25.6 minutes!! ( each episode would be 100 mb)

    This function is to get more sparse data set. It is created to make loading time from pickle into memory fast
    :param d:  same as prepare_dataset
    :param interval: same as prepare_dataset, 120 seconds. it is also for performance
    :param len_observation: Instead of 120 seconds, taking 60 seconds is just for performance
    :param save_dir: root directory where pickle will save
    :return: same as prepare_dataset
    """
    current_date = d['meta']['date']
    current_ticker = d['meta']['ticker']

    print(current_date, current_ticker, 'start')

    c_start = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]),
                                int(current_date[6:8]), 9, 5)  # 9hr 5min 0sec, start time
    c_end = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]),
                              int(current_date[6:8]), 15, 20)  # 15hr 20min 0sec, finish time
    c_rng_ts = pd.date_range(start=c_start, end=c_end,
                                    freq='S')  # range between c_start and c_end saving each seconds' data

    max_idx = len(c_rng_ts) - 1

    x_2d = []  # orderbook
    x_1d = []  # transactions
    x_1d_left_time = []  # left time
    y_1d = []  # width

    for i, s in enumerate(c_rng_ts):

        if i % interval != 0:
            continue

        sys.stdout.write(
            "\rloading data from ticker %s" % current_ticker + ", yyyymmdd %s" % current_date + "  %s" % s + " 완료")
        sys.stdout.flush()

        d_x2d = deque(maxlen=len_sequence_secs)
        d_x1d = deque(maxlen=len_sequence_secs)

        if c_rng_ts[max_idx] < s + len_sequence_secs or i >= max_idx:
            break
        elif s - len_sequence_secs < c_rng_ts[0]:
            continue
        else:
            # first_quote = d['quote'].loc[s]
            # first_order = d['order'].loc[s]
            # threshold = 0.33

            # assemble observation for len_observation
            for i in reversed(range(len_sequence_secs)):
                d_x2d.append(d['order'].loc[s-i])
                d_x1d.append(d['quote'].loc[s-i])

            # calculate min value
            min_price = 0
            for j in range(len_sequence_secs):
                if j == 0:
                    # price_at_signal is the price when the current stock received signal
                    price_at_signal = d['quote'].loc[c_rng_ts[i+j]]['Price(last excuted)']
                    min_price = price_at_signal
                else:
                    price = d['quote'].loc[c_rng_ts[i+j]]['Price(last excuted)']
                    if min_price > price:
                        min_price = price

            x_2d.append(np.array(d_x2d))
            x_1d.append(np.array(d_x1d))
            x_1d_left_time.append(len_sequence_secs)
            y_1d.append(price_at_signal - min_price)

    pickle_name = save_dir + os.path.sep + current_date + '_' + current_ticker + '.pickle'
    f = open(pickle_name, 'wb')
    pickle.dump([x_2d, x_1d, x_1d_left_time, y_1d], f)
    f.close()

    sys.stdout.write("\r")
    sys.stdout.flush()


def main():
    save_dir = 'pickles'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    interval = 601
    len_sequence_secs = 120
    max_secs = 90
    threshold = 0.33

    prepare_datasets(interval, len_sequence_secs, save_dir, max_secs, threshold)


main()