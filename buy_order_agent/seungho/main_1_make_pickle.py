import os
import sys
newPath = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))) + os.path.sep + 'trading-gym'
sys.path.append(newPath)

from gym_core import ioutil
from collections import deque
import pandas as pd
import datetime
import numpy as np
import pickle
import random


def prepare_datasets(interval=120, len_sequence_secs=120, save_dir='pickles'):
    l = ioutil.load_data_from_directory('0')
    for li in l:
        prepare_dataset(li, interval, len_sequence_secs, save_dir)


def prepare_dataset(d, interval, len_sequence_secs, save_dir):
    current_date = d['meta']['date']
    current_ticker = d['meta']['ticker']

    # 시작 년-월-일 09시 05분
    c_start = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]), int(current_date[6:8]), 9, 5)
    # 종료 년-월-일 15시 20분
    c_end = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]), int(current_date[6:8]), 15, 20)
    # 시작 ~ 종료 까지 1초단위로 배열 생성
    c_rng_timestamp = pd.date_range(start=c_start, end=c_end, freq='S')

    threshold = 0.33

    max_idx = len(c_rng_timestamp) - 1

    x_2d = []
    x_1d = []
    x_1d_left_time = []
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

        # SSA 에서 보내주는 남은 시간 랜덤 생성.
        left_secs = random.randint(1, len_sequence_secs)

        # bsa_elapsed_secs : BSA 에서 시그널을 보낸 후 경과한 시간.
        bsa_elapsed_secs = len_sequence_secs - left_secs

        try:
            # first_quote = d['quote'].loc[s]
            # first_order = d['order'].loc[s]

            price = d['quote'].loc[c_rng_timestamp[i]]['Price(last excuted)']
        except KeyError as e:
            print('찾을 수 없는 key 값이 있습니다.', current_ticker, e)
            continue

        # elapsed_secs : SSA 에서 시그널을 받고 경과한 시간
        for elapsed_secs in range(0, left_secs):
            # 너무 데이터가 많아서 0.5% 만 저장.
            if random.random() > 0.001:
                continue

            # 장 시작보다 전에 시그널이 왔으면 skip
            if i < bsa_elapsed_secs + elapsed_secs:
                continue

            # BSA 에서 시그널을 줄 수 있는 시간 이후에 시그널이 왔으면 skip
            if len(c_rng_timestamp) - len_sequence_secs < i - bsa_elapsed_secs:
                continue

            # assemble observation for len_observation
            for i in reversed(range(len_sequence_secs)):
                d_x2d.append(d['order'].loc[s-i])
                d_x1d.append(d['quote'].loc[s-i])

            try:
                price_at_signal = d['quote'].loc[c_rng_timestamp[i - elapsed_secs]]['Price(last excuted)']
            except KeyError as e:
                print('찾을 수 없는 key 값이 있습니다.', current_ticker, e)
                break

            # SOA 가 파는 시점의 X, Y 데이터
            # 시그널 받은 시점의 남은 시간
            x_1d_left_time.append(left_secs)
            # x_1d_second.append(s)
            # order
            x_2d.append(np.array(d_x2d))
            # quote
            x_1d.append(np.array(d_x1d))
            y_1d.append(price - price_at_signal - threshold)

    pickle_name = save_dir + os.path.sep + current_date + '_' + current_ticker + '.pickle'
    f = open(pickle_name, 'wb')
    pickle.dump([x_2d, x_1d, x_1d_left_time, y_1d], f)
    f.close()


def main():
    save_dir = 'pickles'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    prepare_datasets(interval=120, save_dir=save_dir)


main()