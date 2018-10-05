
import pickle

path = "D:/98_myway/study/git/trading-agent/buy_signal_agent/ds/"
pickle_names = [
                'bsa_model_history_e70_b10_n100',
                'bsa_model_history_e70_b10_n120',
                'bsa_model_history_e70_b10_n150',
                'bsa_model_history_e70_b20_n100',
                'bsa_model_history_e100_b10_n70'
                ]

for pickle_name in pickle_names:
    f = open(path + pickle_name, 'rb')
    d = pickle.load(f)  # d[data_type][second] : mapobject!!
    f.close()
    print('loss: {}, mae: {}, mape: {}, mean_pred: {}, theil_u: {}, r: {}'
          .format(d['loss'][-1], d['mean_absolute_error'][-1], d['mean_absolute_percentage_error'][-1],
                  d['mean_pred'][-1], d['theil_u'][-1], d['r'][-1]))

print("------------------------------------------------------------------------------------------------------------")
pickle_names = [
                'bsa_evaluate_model_history_e70_b10_n100',
                'bsa_evaluate_model_history_e70_b10_n120',
                'bsa_evaluate_model_history_e70_b10_n150',
                'bsa_evaluate_model_history_e70_b20_n100',
                'bsa_evaluate_model_history_e100_b10_n70'
                ]

for pickle_name in pickle_names:
    f = open(path + pickle_name, 'rb')
    d = pickle.load(f)  # d[data_type][second] : mapobject!!
    f.close()
    print('mae: {}, mape: {}, mean_pred: {}, theil_u: {}, r: {}'.format(d[2], d[3], d[4], d[5], d[6]))
