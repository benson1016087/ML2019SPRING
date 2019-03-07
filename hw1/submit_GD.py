import sys
import numpy as np
import csv

data = np.genfromtxt(sys.argv[1], delimiter = ',', encoding = 'big5')
data = np.nan_to_num(data)

data = np.delete(data, [0, 1], axis = 1)

#generate X and Y
sz = data.shape[0]
tmp = data[0:18][:]
X = np.transpose(tmp).reshape(1, 18*9)
for i in range(18, sz, 18):
	tmp = np.transpose(data[i:i+18][:]).reshape(1, -1)
	X = np.vstack((X, tmp))
X = np.hstack((np.ones(X.shape[0]).reshape([X.shape[0], 1]), X))

w = np.load("hw1.npy")

y = X.dot(w)
f = open(sys.argv[2], 'w')
f.write('id,value\n')
for i in range(y.shape[0]):
	str_in = 'id_'+str(i)+','+str(float(y[i]))+'\n' #normal
	#str_in = 'id_'+str(i)+','+str(max(round(float(y[i]), ), 0))+'\n' #use int
	f.write(str_in)


