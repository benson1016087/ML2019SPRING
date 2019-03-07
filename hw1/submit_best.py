import sys
import numpy as np
import csv

data = np.genfromtxt(sys.argv[1], delimiter = ',', encoding = 'big5')
data = np.nan_to_num(data)

data = np.delete(data, [0, 1], axis = 1)

#generate X and Y
sz = data.shape[0]
tmp = data[0:18, :]
tmp = np.delete(np.transpose(tmp), [14, 15, 16, 17], axis = 1)
for i in range(9):
	for j in range(tmp.shape[1]):
		if tmp[i][j] < 0:
			if i==0:
				tmp[i][j] = tmp[i][j+1]
			elif i==8:
				tmp[i][j] = tmp[i][j-1]
			else:
				tmp[i][j] = (tmp[i-1][j] + tmp[i+1][j])/2

X = tmp.reshape(1, -1)
for i in range(18, sz, 18):
	tmp = data[i:i+18, :]
	tmp = np.delete(np.transpose(tmp), [14, 15, 16, 17], axis = 1)
	tmp = tmp.reshape(1, -1)
	X = np.vstack((X, tmp))
X = np.hstack((np.ones(X.shape[0]).reshape([X.shape[0], 1]), X))

w = np.load("hw1_best.npy")

y = X.dot(w)
f = open(sys.argv[2], 'w')
f.write('id,value\n')
for i in range(y.shape[0]):
	str_in = 'id_'+str(i)+','+str(max(float(y[i]), 0))+'\n' #normal
	#str_in = 'id_'+str(i)+','+str(max(round(float(y[i]), ), 0))+'\n' #use int
	f.write(str_in)


