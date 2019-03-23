import numpy as np
import pandas as pd
import sys

df_X = pd.read_csv(sys.argv[1])
df_Y = pd.read_csv(sys.argv[2])
df_special = df_X[['age', 'capital_gain', 'capital_loss', 'hours_per_week']]

#generate X and Y (without bias adding in x_0)
X = df_X.values
spec = df_special.values
X = np.hstack((X, spec ** 2, spec ** 3, spec ** 4, spec ** 5, spec ** 6, spec ** 7, spec ** 8, spec ** 9, spec ** 10))
mn = np.mean(X, axis = 0)
vr = np.var(X, axis = 0)
#np.save('mn_dim10.npy', mn)
#np.save('vr_dim10.npy', vr)
X = (X - mn) / np.sqrt(vr)
X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
Y = df_Y.values

#some functions	
def sigmoid(wx):
	return (1 / ( 1 + np.exp(-wx)))

def Err(w, xdata, ydata):
	yp = (sigmoid(xdata.dot(w)) > 0.5)
	cor = np.sum(yp == ydata)
	return (cor / ydata.shape[0])

#some constant variables
eta = 0.02
iteration = 50000
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
lamda = 0.001

#record 
record_vali_E_in = []
record_vali_E_out = []

#doing logistic regression
print('start training:\neta = ', eta, ',iteration = ', iteration, ',beta = ', beta1, beta2, ',lamda = ', lamda)
vali_data_num = X.shape[0] // 5
vali_time = 4
for part in range(vali_time): #doing cross validation
	#set variables
	X_test = X[part*vali_data_num:(part+1)*vali_data_num, :]
	Y_test = Y[part*vali_data_num:(part+1)*vali_data_num]
	X_train = np.vstack((X[:part*vali_data_num, :], X[(part+1)*vali_data_num:, :]))
	Y_train = np.vstack((Y[:part*vali_data_num], Y[(part+1)*vali_data_num:]))
	m = np.zeros([X_train.shape[1], 1])
	v = np.zeros([X_train.shape[1], 1])
	w = np.zeros([X_train.shape[1], 1])
	#using adam
	for ti in range(iteration):
		deltaw = -2 * X_train.T.dot((Y_train - sigmoid(X_train.dot(w)))) + 2 * lamda
		m = beta1 * m + (1 - beta1) * deltaw
		v = beta2 * v + (1 - beta2) * (deltaw ** 2)
		m_bar = m / (1 - beta1)
		v_bar = v / (1 - beta2)
		w = w - eta * m_bar / (np.sqrt(v_bar) + epsilon)
		if (ti+1) % 1000 == 0:
			E_in = Err(w, X_train, Y_train)
			print('part',part,': ',ti+1,'tiems done, correct_in = ', E_in)
	E_in = Err(w, X_train, Y_train)
	E_out = Err(w, X_test, Y_test)
	print("correct_in: ",E_in, "; correct_out", E_out)
	record_vali_E_in.append(E_in)
	record_vali_E_out.append(E_out)

#print the ending result
print('end of validation:\neta = ', eta, ',iteration = ', iteration, ',beta = ', beta1, beta2, ',lamda = ', lamda)
for i in range(vali_time):
	print('part ',i+1,': ','correct_in = ',record_vali_E_in[i], '; correct_out = ',record_vali_E_out[i])

#main test 
m = np.zeros([X.shape[1], 1])
v = np.zeros([X.shape[1], 1])
w = np.zeros([X.shape[1], 1])
for ti in range(iteration):
	deltaw = -2 * X.T.dot((Y - sigmoid(X.dot(w))))
	m = beta1 * m + (1 - beta1) * deltaw
	v = beta2 * v + (1 - beta2) * (deltaw ** 2) + 2 * lamda
	m_bar = m / (1 - beta1)
	v_bar = v / (1 - beta2)
	w = w - eta * m_bar / (np.sqrt(v_bar) + epsilon)
	if (ti+1) % 1000 == 0:
		E_in = Err(w, X, Y)
		print('main:',ti+1,'tiems done, E_in = ', E_in)
print('end training:\neta = ', eta, ',iteration = ', iteration, ',beta = ', beta1, beta2, ',lamda = ', lamda)
np.save(sys.argv[3], w)
