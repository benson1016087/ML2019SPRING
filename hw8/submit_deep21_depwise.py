import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LeakyReLU, DepthwiseConv2D
from keras import backend as K
import tensorflow as tf
import pandas as pd
import sys, os

data = pd.read_csv(sys.argv[1])
data_X = data['feature'].str.split(' ', expand = True)
X = data_X.values.astype('float')
X = (X/255).reshape(-1, 48, 48, 1)


model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (48, 48, 1), padding = 'same'))
model.add(BatchNormalization())
model.add(DepthwiseConv2D(kernel_size = (3, 3), padding = 'same'))
model.add(Conv2D(filters = 32, kernel_size = (1, 1), padding = 'same'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(DepthwiseConv2D(kernel_size = (3, 3), padding = 'same'))
model.add(Conv2D(filters = 64, kernel_size = (1, 1), padding = 'same'))
model.add(BatchNormalization())
model.add(DepthwiseConv2D(kernel_size = (3, 3), padding = 'same'))
model.add(Conv2D(filters = 64, kernel_size = (1, 1), padding = 'same'))
model.add(LeakyReLU())
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(DepthwiseConv2D(kernel_size = (3, 3), padding = 'same'))
model.add(Conv2D(filters = 128, kernel_size = (1, 1), padding = 'same'))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(DepthwiseConv2D(kernel_size = (3, 3), padding = 'same'))
model.add(Conv2D(filters = 128, kernel_size = (1, 1), padding = 'same'))
model.add(LeakyReLU())
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(units = 7, activation = 'softmax'))

w = np.load(sys.argv[2], allow_pickle=True)
model.set_weights(w)

pred = model.predict(X)
opt = np.argmax(pred, axis = 1)

f = open(sys.argv[3], 'w')
wt_str = 'id,label\n'
f.write(wt_str)
for i in range(opt.shape[0]):
	wt_str = str(i)+','+str(int(opt[i]))+'\n'
	f.write(wt_str)