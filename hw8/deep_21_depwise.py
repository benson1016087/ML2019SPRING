import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, LeakyReLU, DepthwiseConv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
import sys, os
import pandas as pd

data = pd.read_csv(sys.argv[1])
Y = data['label'].values.reshape(-1, 1)
data_X = data['feature'].str.split(' ', expand = True)
X = data_X.values.astype('float')

#generate X and Y
X /= 255
Y = np.hstack((Y == 0, Y == 1, Y == 2, Y == 3, Y == 4, Y == 5, Y == 6))

#cutting validation set
X_train = X[:-3000, :].reshape(-1, 48, 48, 1)
X_val = X[-3000:, :].reshape(-1, 48, 48, 1)
Y_train = Y[:-3000, :]
Y_val = Y[-3000:, :]

# building the CNN model 
def building_convolution_model():
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
	return model

# building DNN model
def fully_connected_model(model):
	model.add(Dense(units = 7, activation = 'softmax'))
	return model

# Generator variable
img_generator = ImageDataGenerator(rotation_range = 25, horizontal_flip = True, width_shift_range = 0.1, height_shift_range = 0.1)

# Main 
model = building_convolution_model()
model = fully_connected_model(model)
print(model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

filepath = sys.argv[2]
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(img_generator.flow(X_train, Y_train, batch_size = 128), steps_per_epoch = X_train.shape[0] / 128, epochs = 400, callbacks = callbacks_list, validation_data = (X_val, Y_val))





