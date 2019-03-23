import numpy as np
import sys
import pandas as pd

df_X = pd.read_csv(sys.argv[1])
df_Y = pd.read_csv(sys.argv[2])

#generate data
data = np.hstack((df_X.values, df_Y.values))
np.random.shuffle(data)
data_vali = data[-3000:, :]
data = data[:-3000, :]

#classify the traning data
data_0 = np.zeros(data.shape[1])
data_1 = np.zeros(data.shape[1])

cnt = 1
for i in data:
	if i[-1] == 0:
		data_0 = np.vstack((data_0, i))
	elif i[-1] == 1:
		data_1 = np.vstack((data_1, i))
	else: 
		print('something err')

print('end dealing data')

data_0 = np.delete(data_0, 0, axis = 0)
data_1 = np.delete(data_1, 0, axis = 0)

X_0 = data_0[:, :-1]
X_1 = data_1[:, :-1]

#calculate the mn and cov
mean_0 = np.mean(X_0, axis = 0)
mean_1 = np.mean(X_1, axis = 0)

cov_0 = np.cov(X_0.T)
cov_1 = np.cov(X_1.T)
print('end calculating mean/cov')

num_0 = mean_0.shape[0]
num_1 = mean_1.shape[0]
dim = X_0.shape[1]
cov_total = num_0/(num_0+num_1)*cov_0 + num_1/(num_0+num_1)*cov_1

w = (mean_0 - mean_1).T.dot(np.linalg.inv(cov_total))
b = -0.5 * mean_0.dot(np.linalg.inv(cov_total)).dot(mean_0) + 0.5 * mean_1.T.dot(np.linalg.inv(cov_total)).dot(mean_1) + np.log(num_0 / num_1)

np.save('w_generative.npy', w)
np.save('b_generative.npy', b)








