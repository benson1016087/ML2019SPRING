import numpy as np
import sys
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as GBC
import pickle

X_test = pd.read_csv(sys.argv[1]).values

model = pickle.load(open(sys.argv[3], 'rb'))
opt = model.predict(X_test)

f = open(sys.argv[2], 'w')
wt_str = 'id,label\n'
f.write(wt_str)
for i in range(opt.shape[0]):
	wt_str = str(i+1)+','+str(int(opt[i]))+'\n'
	f.write(wt_str)

