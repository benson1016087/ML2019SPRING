import sys
import numpy as np

data = np.genfromtxt(sys.argv[1], delimiter = ',', encoding = 'big5')
data = np.nan_to_num(data)

data = np.delete(data, [0, 1, 2], axis = 1)
data = np.delete(data, 0, axis = 0)

#generate X and Y
feature = data[0:18][:]
pm = data[9][:]
sz = data.shape[0]
for i in range(18, sz, 18):
	feature = np.hstack((feature, data[i:i+18][:]))
	pm = np.hstack((pm, data[i+9][:]))
feature = np.transpose(feature)
feature = np.delete(feature, [14, 15, 16, 17], axis = 1)
for i in range(feature.shape[0]): # 8th, 9th are PM10 and PM2.5
	for j in range(feature.shape[1]):
		if feature[i][j] < 0:
			feature[i][j] = (feature[i-1][j] + feature[i+1][j])/2


X = feature[0:9][:].reshape(1, -1)
Y = pm[9]
for i in range(1, feature.shape[0]-9):
	if i%240>230 and i%240<=239:
		continue;
	X = np.vstack((X, feature[i:i+9][:].reshape(1, -1)))
	Y = np.vstack((Y, pm[i+9]))
X = np.hstack((np.ones(X.shape[0]).reshape([X.shape[0], 1]), X))
datasize = X.shape[0]
print("X.shape = ",X.shape)

#initialize the variables
w = np.ones([X.shape[1], 1])
eta = 10
ada = np.zeros([X.shape[1], 1])
iteration = 200000
lamda = 0

#calculate E_in 
def E_in():
	err = Y - X.dot(w)
	error = np.transpose(err).dot(err)
	return (error / X.shape[0]) ** 0.5


#training
print('start training:\nlamda = ', lamda, ',eta = ', eta, ',iteration = ', iteration)
for ti in range(iteration):
	delta_w = -2 * np.transpose(X).dot(Y - X.dot(w)) + lamda * w
	ada += delta_w ** 2
	w = w - eta * delta_w / np.sqrt(ada)
	if (ti+1)%1000 == 0:
		error = E_in()
		print(ti+1,'tiems done, E_in = ', error)

np.save(sys.argv[2], w)
print('end of train:\nlamda = ', lamda, ',eta = ', eta, ',iteration = ', iteration)

