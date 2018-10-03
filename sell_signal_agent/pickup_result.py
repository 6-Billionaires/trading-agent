
import pickle

pickle_names = [
                '/Users/doni/99_study/git/trading-agent/sell_signal_agent/ssa_model_history_e75_b70_n75',
                '/Users/doni/99_study/git/trading-agent/sell_signal_agent/ssa_model_history_e50_b70_n125',
                '/Users/doni/99_study/git/trading-agent/sell_signal_agent/ssa_model_history_e100_b30_n175',
                '/Users/doni/99_study/git/trading-agent/sell_signal_agent/ssa_model_history_e100_b50_n175',
                '/Users/doni/99_study/git/trading-agent/sell_signal_agent/ssa_model_history_e100_b70_n125'
                ]

for pickle_name in pickle_names:
    f = open(pickle_name, 'rb')
    d = pickle.load(f)  # d[data_type][second] : mapobject!!
    f.close()
    print('loss: {}, mae: {}, mape: {}, mean_pred: {}, theil_u: {}, r: {}'
          .format(d['loss'][-1], d['mean_absolute_error'][-1], d['mean_absolute_percentage_error'][-1], d['mean_pred'][-1], d['theil_u'][-1], d['r'][-1]))