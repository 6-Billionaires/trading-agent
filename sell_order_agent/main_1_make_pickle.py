import os
import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-training", "--training", help="turn on training mode", action="store_true")
parser.add_argument("-import-gym", "--import-gym",help="import trading gym", action="store_true")
parser.add_argument("-gym-dir", "--gym-dir", type=str, help="import trading gym")
parser.add_argument("-project-dir", "--project-dir", type=str, help="import project home")
args = parser.parse_args()

if args.import_gym:
    sys.path.insert(0, args.gym_dir)
    sys.path.insert(1, args.project_dir)

from gym_core import ioutil
from collections import deque
import pandas as pd
import datetime
import numpy as np
import pickle
import random

import config

if args.training:
    csv_dir = config.SOA_PARAMS['CSV_DIR_FOR_CREATING_PICKLE_TRAINING']
    save_dir = config.SOA_PARAMS['PICKLE_DIR_FOR_TRAINING']
else:
    csv_dir = config.SOA_PARAMS['CSV_DIR_FOR_CREATING_PICKLE_TEST']
    save_dir = config.SOA_PARAMS['PICKLE_DIR_FOR_TEST']


def prepare_datasets(load_csv_dir, interval=120, len_sequence_secs=120, save_dir='pickles'):
    l = ioutil.load_data_from_directory(load_csv_dir, '0')
    for li in l:
        prepare_dataset(li, interval, len_sequence_secs, save_dir)

def prepare_dataset(d, interval, len_sequence_secs, save_dir):
    current_date    = d['meta']['date']
    current_ticker  = d['meta']['ticker']

    # start-time : YYYY-MM-DD 09:05
    c_start = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]), int(current_date[6:8]), 9, 5)
    # end-time : YYYY-MM-DD 15:20
    c_end = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]), int(current_date[6:8]), 15, 20)
    # create array for every second from start-time to end-time
    c_rng_timestamp = pd.date_range(start=c_start, end=c_end, freq='S')

    threshold = 0.33

    max_idx = len(c_rng_timestamp) - 1

    x_2d = []
    x_1d = []
    x_1d_left_time = []
    x_1d_elapsed_time = []
    y_1d = []

    for i, s in enumerate(c_rng_timestamp):
        if c_rng_timestamp[max_idx] < s + len_sequence_secs or i >= max_idx:
            break
        elif s - len_sequence_secs < c_rng_timestamp[0]:
            continue
        elif i % interval != 0:
            continue

        d_x2d = deque(maxlen=len_sequence_secs)
        d_x1d = deque(maxlen=len_sequence_secs)

        # create randomly the remaining time sent from SSA.
        left_secs = random.randint(1, len_sequence_secs)

        # bsa_elapsed_secs : elapsed time after sending signal from BSA
        bsa_elapsed_secs = len_sequence_secs - left_secs

        try:
            price = d['quote'].loc[c_rng_timestamp[i]]['Price(last executed)']
        except KeyError as e:
            print('cannot find the key value.', current_ticker, e)
            continue

        # elapsed_secs : elapsed time after receiving signal from SSA
        for elapsed_secs in range(0, left_secs):
            # store only 0.5% because data is too much
            if random.random() > 0.05:
                continue

            # skip if the signal occurs before stock market starts
            if i < bsa_elapsed_secs + elapsed_secs:
                continue

            # skip if signal is received after the time BSA can send signal
            if len(c_rng_timestamp) - len_sequence_secs < i - bsa_elapsed_secs:
                continue

            try:
                # assemble observation for len_observation
                for i in reversed(range(len_sequence_secs)):
                    d_x2d.append(d['order'].loc[s-i])
                    d_x1d.append(d['quote'].loc[s-i])

                price_at_signal = d['quote'].loc[c_rng_timestamp[i - elapsed_secs]]['Price(last executed)']
            except KeyError as e:
                print('cannot find the key value.', current_ticker, e)
                break

            # X, Y data at the point of selling by SOA
            # the remaining time at the point of receiving signal
            x_1d_left_time.append(left_secs)
            # elapsed time after receiving signal
            x_1d_elapsed_time.append(elapsed_secs)
            # x_1d_second.append(s)
            # order
            x_2d.append(np.array(d_x2d))
            # quote
            x_1d.append(np.array(d_x1d))
            y_1d.append(price - price_at_signal - threshold)

    # create only when data exists
    if len(x_2d) > 0:
        pickle_name = save_dir + os.path.sep + current_date + '_' + current_ticker + '.pickle'
        f = open(pickle_name, 'wb')
        pickle.dump([x_2d, x_1d, x_1d_left_time, x_1d_elapsed_time, y_1d], f)
        f.close()
        print('{} file is created.'.format(pickle_name))

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

prepare_datasets(load_csv_dir=csv_dir, interval=120, save_dir=save_dir)
