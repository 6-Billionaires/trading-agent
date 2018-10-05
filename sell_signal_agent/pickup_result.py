
import pickle

path = "D:/98_myway/study/git/trading-agent/sell_signal_agent/"
pickle_names = [
                'ssa_model_history_e75_b70_n75',
                'ssa_model_history_e50_b70_n125',
                'ssa_model_history_e100_b30_n175',
                'ssa_model_history_e100_b50_n175',
                'ssa_model_history_e100_b70_n125'
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
                'ssa_evaluate_model_history_e75_b70_n75',
                'ssa_evaluate_model_history_e50_b70_n125',
                'ssa_evaluate_model_history_e100_b30_n175',
                'ssa_evaluate_model_history_e100_b50_n175',
                'ssa_evaluate_model_history_e100_b70_n125'
                ]

for pickle_name in pickle_names:
    f = open(path + pickle_name, 'rb')
    d = pickle.load(f)  # d[data_type][second] : mapobject!!
    f.close()
    print('mae: {}, mape: {}, mean_pred: {}, theil_u: {}, r: {}'.format(d[2], d[3], d[4], d[5], d[6]))
