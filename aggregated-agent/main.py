import keras
from keras.models import Model
from keras.layers import Input, Dense, Multiply

# agent: buy signal
input_buy_signal = Input(shape=(None, 10))
buy_signal = Dense(2)(input_buy_signal)

# agent: buy order
input_buy_order = Input(shape=(None, 10))
buy_signal_action = Input(shape=(None, 1)) # 0: stop / 1: go
c_buy_order = keras.layers.Concatenate()([input_buy_signal, buy_signal_action])
h_buy_order = Dense(2)(c_buy_order)
# output: Q-value, 0 q-value if buy_signal_action = 0
buy_order = Multiply()([h_buy_order, buy_signal_action])

# agent: sell signal
input_sell_signal = Input(shape=(None, 10))
buy_order_action = Input(shape=(None, 1))# 0: stop / 1: go
c_sell_signal = keras.layers.Concatenate()([input_sell_signal, buy_order_action])
h_sell_signal = Dense(2)(c_sell_signal)
sell_signal = Multiply()([h_sell_signal, buy_order_action])

# agent: sell order
input_sell_order = Input(shape=(None, 10))
sell_signal_action = Input(shape=(None, 1))# 0: stop / 1: go
c_sell_signal = keras.layers.Concatenate()([input_sell_signal, sell_signal_action])
h_sell_order = Dense(2)(c_sell_signal)
sell_order = Multiply()([h_sell_order, sell_signal_action])

agent = Model([input_buy_signal, input_buy_order, input_sell_signal, input_sell_order,
               buy_signal_action, buy_order_action, sell_signal_action],
              [buy_signal, buy_order, sell_signal, sell_order])

agent.summary()