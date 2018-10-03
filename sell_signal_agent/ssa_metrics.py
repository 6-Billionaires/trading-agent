
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def theil_u(y_true, y_pred):
    up = K.sqrt(K.mean(K.square(y_true - y_pred)))
    bottom = K.sqrt(K.mean(K.square(y_true))) + K.sqrt(K.mean(K.square(y_pred)))
    return up / bottom

def r(y_true, y_pred):
    mean_y_true = K.mean(y_true)
    mean_y_pred = K.mean(y_pred)

    up = K.sum((y_true - mean_y_true) * (y_pred - mean_y_pred))
    bottom = K.sqrt(K.sum(K.square(y_true - mean_y_true)) * K.sum(K.square(y_pred - mean_y_pred)))

    return up / bottom


dict_to_plot = {
    'LOSS':'loss',
    'MAE' : 'mean_absolute_error',
    'MAPE' : 'mean_absolute_percentage_error',
    'Mean_Pred' : 'mean_pred',
    'Corr' : 'r',
    "Theil_U" : 'theil_u'
}