
import os
import sys
newPath = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))) + '/trading-gym'
sys.path.append(newPath)


from gym_core import ioutil  # file i/o to load stock csv files
from collections import deque
import config
import pandas as pd
import datetime
import numpy as np
import pickle
import os
import random

"""
previously,  I gave secs as 120. but like iljoo said, it needs to be 120.
in other agents, it can changes but you don't have to read every time seconds periods changes
just read maximum periods of data and reuse it   
"""
def prepare_datasets(is_spare_dataset=False, interval=120, len_observation=60, len_sequence_secs=120, save_dir=''):
    """
    main coordinate fucntion to create pickle
    :param is_spare_dataset: if true, it uses not prepare_dataset function, but prepares_spare_dataset function.
    :param interval: same as prepare_sparse_dataset
    :param len_observation: only used if is_spare_dataset is true
    :param len_sequence_secs: same as prepare_sparse_dataset
    :param save_dir: root directory where pickle will save
    :return:
    """
    # l = ioutil.load_data_from_directory('0', max_n_episode=1) # episode type
    l = ioutil.load_data_from_directory('0')  # episode type
    for li in l:
        if is_spare_dataset:
            prepare_sparse_dataset(li, 120, 120, 60, save_dir)
        else:
            prepare_dataset(li, 1, len_sequence_secs)


def prepare_sparse_dataset(d, interval=120, len_sequence_of_secs=120, len_observation=60, save_dir=''):
    """
    original version
    loading data from ticker 20180403, yyyymmdd 003350 is started.
    executed time :  1538.569629558868 -> 25.6 minutes!! ( each episode would be 100 mb)
    This function is to get more sparse data set. It is created to make loading time from pickle into memory fast
    :param d:  same as prepare_dataset
    :param interval: same as prepare_dataset, 120 seconds. it is also for performance
    :param len_sequence_of_secs:  same as prepare_dataset.
    :param len_observation: Instead of 120 seconds, taking 60 seconds is just for performance
    :param save_dir: root directory where pickle will save
    :return: same as prepare_dataset
    """
    current_date = d['meta']['date']
    current_ticker = d['meta']['ticker']

    c_start = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]),
                                int(current_date[6:8]), 9, 5)  # 9hr 5min 0sec, start time
    c_end = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]),
                              int(current_date[6:8]), 15, 20)  # 15hr 20min 0sec, finish time
    c_rng_ts = pd.date_range(start=c_start, end=c_end,
                                    freq=str(interval)+'S')  # range between c_start and c_end saving each seconds' data

    max_idx = len(c_rng_ts) - 1

    x_2d = []  # orderbook
    x_1d = []  # transactions
    y_1d = []  # width

    # 경과시간
    elapsed_time = []
    # 남은시간
    left_time = []

    for i, s in enumerate(c_rng_ts):
        # BOA 에서 보내주는 남은 시간 랜덤 생성.
        left_secs = random.randint(1, max_secs)

        # SSA 에서 시그널 발생 하는데 까지 걸린 시간을 랜덤 생성.
        elapsed_secs = random.randint(0, left_secs)

        d_x2d = deque(maxlen=len_observation)
        d_x1d = deque(maxlen=len_observation)

        if c_rng_ts[max_idx] < s + pd.Timedelta(seconds=len_sequence_of_secs) or i >= max_idx:
        # if c_rng_ts[max_idx] < s + len_sequence_of_secs or i >= max_idx:
            break
        elif s - pd.Timedelta(seconds=len_observation) < c_rng_ts[0]:
        # elif s - len_observation < c_rng_ts[0]:
            continue
        else:
            width = 0
            threshold = 0.33

            # assemble observation for len_observation
            for i in reversed(range(len_observation)):
                d_x2d.append(d['order'].loc[s-pd.Timedelta(seconds=i)])
                d_x1d.append(d['quote'].loc[s-pd.Timedelta(seconds=i)])
                # d_x2d.append(d['order'].loc[s-i])
                # d_x1d.append(d['quote'].loc[s-i])

            # price_at_signal is the price when the current stock received signal
            price_at_signal = d['quote'].loc[c_rng_ts[i-elapsed_secs]]['Price(last excuted)']
            price = d['quote'].loc[c_rng_ts[i]]['Price(last excuted)']
            gap = price_at_signal - price - threshold
            width += gap

            x_2d.append(np.array(d_x2d))
            x_1d.append(np.array(d_x1d))
            left_time.append(left_secs)
            elapsed_time.append(elapsed_secs)
            y_1d.append(width)

    pickle_name = save_dir + os.path.sep + current_date + '_' + current_ticker + '.pickle'
    f = open(pickle_name, 'wb')
    pickle.dump([x_2d, x_1d, elapsed_time, left_time, y_1d], f)
    f.close()


def prepare_dataset(d, interval=1, len_sequence_of_secs=120):
    """
    :param d
        the variable having pickle file data in memory
    :param interval
        a period between previous observation and current observation,
        if it bring data moving 1 second forward, data size is so huge.
    :param len_sequence_of_secs:
        each observation length
    :return:
        nothing, it end up saving list of pickle files as you configured
    """
    current_date = d['meta']['date']
    current_ticker = d['meta']['ticker']

    d_price = deque(maxlen=len_sequence_of_secs)

    c_start = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]),
                                int(current_date[6:8]), 9, 5)  # 9hr 5min 0sec, start time
    c_end = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]),
                              int(current_date[6:8]), 15, 20)  # 15hr 20min 0sec, finish time
    c_rng_ts = pd.date_range(start=c_start, end=c_end,
                                    freq=str(interval)+'S')  # range between c_start and c_end saving each seconds' data

    x_2d = []  # orderbook
    x_1d = []  # transactions
    y_1d = []  # width

    # 경과시간
    elapsed_time = []
    # 남은시간
    left_time = []

    max_idx = len(c_rng_ts) - 1

    for i, s in enumerate(c_rng_ts):
        # BOA 에서 보내주는 남은 시간 랜덤 생성.
        left_secs = random.randint(1, max_secs)
        # BSA 에서 시그널을 보낸 후 경과한 시간.
        bsa_elapsed_secs = max_secs - left_secs
        # SSA 에서 시그널 발생 하는데 까지 걸린 시간을 랜덤 생성.
        elapsed_secs = random.randint(0, left_secs)

        if c_rng_ts[max_idx] < c_rng_ts[i] + len_sequence_of_secs or i >= max_idx:
            break
        elif s - len_sequence_of_secs < c_rng_ts[0]:
            continue
        else:
            first_quote = d['quote'].loc[s]
            first_order = d['order'].loc[s]
            width = 0
            threshold = 0.33

            # calculate width
            for j in range(len_sequence_of_secs):
                if j == 0:
                    # price_at_signal is the price when the current stock received signal
                    price_at_signal = d['quote'].loc[c_rng_ts[i+j]]['Price(last excuted)']
                else:
                    price = d['quote'].loc[c_rng_ts[i+j]]['Price(last excuted)']
                    gap = price_at_signal - price - threshold
                    width += gap

            left_time.append(left_secs)
            elapsed_time.append(elapsed_secs)
            x_2d.append(first_order)
            x_1d.append(first_quote)
            y_1d.append(width)

    pickle_name = current_date + '_' + current_ticker + '.pickle'
    f = open(pickle_name, 'wb')
    pickle.dump([x_2d, x_1d, y_1d], f)
    f.close()


save_dir = 'sparse'
max_secs = 120
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

prepare_datasets(is_spare_dataset=True, save_dir=save_dir)