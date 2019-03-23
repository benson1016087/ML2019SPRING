import numpy as np
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])
df_special = df[['age', 'capital_gain', 'capital_loss', 'hours_per_week']]
X = df.values
spec = df_special.values
X = np.hstack((X, spec ** 2, spec ** 3, spec ** 4, spec ** 5, spec ** 6, spec ** 7, spec ** 8, spec ** 9, spec ** 10))
mn = np.load('mn_dim10_nfnl.npy')
vr = np.load('vr_dim10_nfnl.npy')
X = (X - mn) / np.sqrt(vr)
X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))

w = np.load(sys.argv[3])
Y = X.dot(w)
opt = (Y > 0.5)

f = open(sys.argv[2], 'w')
wt_str = 'id,label\n'
f.write(wt_str)
for i in range(opt.shape[0]):
	wt_str = str(i+1)+','+str(int(opt[i]))+'\n'
	f.write(wt_str)




