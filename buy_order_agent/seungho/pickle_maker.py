from gym_core import tgym
from gym_core import ioutil
from collections import deque
import config
import pandas as pd
import datetime
import numpy as np
import pickle


def prepare_datasets(max_secs=90):
    l = ioutil.load_data_from_directory('0')

    # prepare_dataset(l[0], max_secs)
    for li in l:
        prepare_dataset(li, max_secs)


def prepare_dataset(d, max_secs):
    current_date = d['meta']['date']
    current_ticker = d['meta']['ticker']

    c_start = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]), int(current_date[6:8]), 9, 5)
    c_end = datetime.datetime(int(current_date[0:4]), int(current_date[4:6]), int(current_date[6:8]), 15, 20)
    c_rng_timestamp = pd.date_range(start=c_start, end=c_end, freq='S')

    x_time = []
    # x_2d = []
    # x_1d = []
    x_elapsed_secs = []
    y_1d = []

    price_at_signal_list = []
    time_at_signal_list = []
    time_at_min_price_list = []
    min_price_list = []

    # for i in range(max_secs):
    #     price_at_signal_list.append(d['quote'].loc[c_rng_timestamp[0]]['Price(last excuted)'])

    # first_quote = d['quote'].loc[c_rng_timestamp[0]]
    # first_order = d['order'].loc[c_rng_timestamp[0]]

    # print(c_rng_timestamp)
    for i, s in enumerate(c_rng_timestamp):
        # if i > 3600:
        #     break
        if i % 3600 == 0:
            print(current_date, current_ticker, s)

        try:
            first_quote = d['quote'].loc[s]
            first_order = d['order'].loc[s]
        except KeyError as e:
            print('찾을 수 없는 key 값이 있습니다.', current_ticker, e)

        # 현재 가격 확인 후
        price_at_current = d['quote'].loc[c_rng_timestamp[i]]['Price(last excuted)']

        # max_secs 동안의 최저값을 갱신함
        for index, min_price in enumerate(min_price_list):
            if min_price > price_at_current:
                min_price_list[index] = price_at_current
                time_at_min_price_list[index] = s

        # 신규 데이터 추가
        price_at_signal_list.append(price_at_current)
        time_at_signal_list.append(s)
        time_at_min_price_list.append(s)
        min_price_list.append(price_at_current)

        # x_2d.append(first_order)
        # x_1d.append(first_quote)

        # max_sec 초과한 데이터는 적재 시작
        if len(min_price_list) >= max_secs:
            price_at_signal = price_at_signal_list.pop(0)
            time_at_signal = time_at_signal_list.pop(0)
            elapsed_secs = int((time_at_min_price_list.pop(0) - time_at_signal).total_seconds())
            y = price_at_signal - min_price_list.pop(0)
            x_time.append(time_at_signal)
            x_elapsed_secs.append(elapsed_secs)
            y_1d.append(y)
            # print(time_at_signal, elapsed_secs, y)

    # 남은 데이터 적재
    for index in enumerate(price_at_signal_list):
        price_at_signal = price_at_signal_list.pop(0)
        time_at_signal = time_at_signal_list.pop(0)
        elapsed_secs = int((time_at_min_price_list.pop(0) - time_at_signal).total_seconds())
        y = price_at_signal - min_price_list.pop(0)
        x_time.append(time_at_signal)
        x_elapsed_secs.append(elapsed_secs)
        y_1d.append(y)
        # print(time_at_signal, elapsed_secs, y)

    pickle_name = current_date + '_' + current_ticker + '.pickle'

    # 시간만 저장하도록 변경하는 코드
    f = open('./pickles/' + pickle_name, 'wb')
    pickle.dump([x_time, x_elapsed_secs, y_1d], f)

    f.close()


prepare_datasets()
