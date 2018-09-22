from keras.models import load_model
import numpy as np


def get_sample_data(count):
    start = 0
    ld_x1 = []
    ld_x2 = []
    ld_x3 = []
    ld_y = []
    d1 = []

    for i in range(count):
        d1_2 = np.arange(start, start + 10 * 2 * 120 * 2).reshape([10, 2, 120, 2])
        start += 2 * 10 * 120
        d2 = np.arange(start, start + 11 * 120).reshape([120, 11])
        start += 11 * 120
        d3 = np.arange(start, start + 7).reshape([7])
        start += 7

        ld_x1.append(d1_2)
        ld_x2.append(d2)
        ld_x3.append(d3)

    for j in range(count):
        d1 = np.arange(start, start + 1)
        ld_y.append(d1)

    return np.asarray(ld_x1), np.asarray(ld_x2), np.asarray(ld_x3), np.asarray(ld_y)



# max_len = util.get_maxlen_of_binary_array(120)
# model = build_network(max_len, neurons=100, activation='leaky_relu')
# model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'], )
# model.summary()
print('load model')
model  = load_model('soa_weights_final.h5')
print('model loaded!')
# load weight
#model.load_weights('final_weight.h5f')

x1, x2, x3, y = get_sample_data(10) #for test
print('start evaluate')
scores = model.evaluate({'x1': x1, 'x2': x2, 'x3': x3}, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
