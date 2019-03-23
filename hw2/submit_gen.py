import numpy as np
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
X = df.values

w = np.load('w_generative.npy')
b = np.load('b_generative.npy')

Y_predict = (1 / (1 + np.exp(-(X.dot(w)+b))))
opt = (Y_predict <= 0.5)

f = open(sys.argv[2], 'w')
wt_str = 'id,label\n'
f.write(wt_str)
for i in range(opt.shape[0]):
	wt_str = str(i+1)+','+str(int(opt[i]))+'\n'
	f.write(wt_str)