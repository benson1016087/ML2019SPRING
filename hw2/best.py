import numpy as np
import sys
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as GBC
import pickle

df_X = pd.read_csv(sys.argv[1])
df_Y = pd.read_csv(sys.argv[2])

X = df_X.values
Y = df_Y.values
X_test = pd.read_csv(sys.argv[3]).values

#devide training data and validation data
X_train = X[:-3000]
X_val = X[-3000:]
Y_train = Y[:-3000]
Y_val = Y[-3000:]

#training 
print('start GBC')
model = GBC(n_estimators = 1000, random_state = 6, min_samples_split = 3)
model.fit(X_train, Y_train.reshape(-1, ))

#predict the result
prd = model.predict(X_val)
correct_rate = np.sum(prd.reshape(-1, 1) == Y_val) / Y_val.shape[0]
print('correct_rate =', correct_rate)

#output the result
opt = model.predict(X_test)
f = open(sys.argv[4], 'w')
wt_str = 'id,label\n'
f.write(wt_str)
for i in range(opt.shape[0]):
	wt_str = str(i+1)+','+str(int(opt[i]))+'\n'
	f.write(wt_str)

pickle.dump(model, open(sys.argv[5], 'wb'))