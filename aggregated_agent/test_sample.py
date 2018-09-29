from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


model = Sequential()
model.add(Dense(100, input_dim=3120, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.save('./test.h5')