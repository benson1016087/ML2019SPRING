import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers.core import Dense,Activation
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import sys
import csv

#extract data
data = pd.read_csv(sys.argv[1])
data_X = data['feature'].str.split(' ', expand = True)
X = data_X.values.astype('float')
X = (X/255).reshape(-1, 48, 48, 1)

#predict
pred = (load_model(sys.argv[3])).predict(X)
print('finish 0')
for i in range(4, 8):
	model = load_model(sys.argv[i])
	pred += model.predict(X)
	print('finish', i-3)
opt = np.argmax(pred, axis = 1)
'''
index = np.arange(0, opt.shape[0]).reshape(-1, 1)
print(index.shape, opt.shape)
opt_index = np.hstack((index, opt.reshape(-1, 1)))
df = pd.DataFrame(opt_index) # A is a numpy 2d array
df.to_excel(sys.argv[2], header = "id,label", index = False)
'''

f = open(sys.argv[2], 'w')
wt_str = 'id,label\n'
f.write(wt_str)
for i in range(opt.shape[0]):
	wt_str = str(i)+','+str(int(opt[i]))+'\n'
	f.write(wt_str)
    
print('finish output')
