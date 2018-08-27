import pickle

current_date ='20180420'
current_ticker = '001470'

#x1, x2,  y = get_sample_data(10)
##(60, 11),  (10, 2, 60, 2)

pickle_name = current_date + '_' + current_ticker + '.pickle'
f = open(pickle_name, 'rb')
a = pickle.load(f)
f.close()

print(a[0][0]['BuyHoga1'])
print(a[0][0][0])
print(a[0][0][1])
print(a[0][0][2])
print(a[0][0][3])
