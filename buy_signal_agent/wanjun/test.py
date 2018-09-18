import os
import sys

newPath = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))+ '\\trading-gym'
sys.path.append(newPath)

from gym_core import tgym
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dropout, Concatenate, Embedding
from keras.optimizers import Adam
from keras import models

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.processors import MultiInputProcessor

import logging
import time

from rl.core import Processor
from collections import deque
from rl.callbacks import Callback

from keras.preprocessing import sequence
from keras.datasets import reuters

(x, y), (_, _) = reuters.load_data(num_words=1000)
x = sequence.pad_sequences(x, maxlen=120)
model = Sequential()
model.add(Embedding(120, 128, input_length=1000))
model.add(Dropout(0.2))
model.add(Conv1D(256,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(Dense(46, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y)
# i = Input(shape=(1, 120))
# h = Conv1D(32, kernel_size=4, activation='relu')
# # h = MaxPooling1D(pool_size=4)(h)
# h = Flatten()(h)
# h = Activation('relu')(Dense(1)(h))
# model = models.Model(i, h)
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.summary()
#
# model.fit(x=x, y=y, batch_size=32)