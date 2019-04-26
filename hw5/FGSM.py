from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from skimage.io import imread, imsave
import keras.backend as K
import numpy as np
import os, sys

input_file = sys.argv[1]
ouput_file = sys.argv[2]
model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
mean = [103.939, 116.779, 123.68]

label = K.placeholder(dtype='int32')
target = K.one_hot(label, 1000)
loss = K.categorical_crossentropy(target, model.output)
grads = K.gradients(loss, model.input)[0]
fn = K.function([model.input, K.learning_phase(), label], [grads])

for i in range(200):
	# loading image
	index = str(i).zfill(3)
	img = imread(os.path.join(input_file, index + '.png'))
	img = img.reshape(-1, 224, 224, 3)
	img = preprocess_input(img)
	
	# predict
	Y_pred = model.predict(img)
	X_label = np.argmax(Y_pred)

	#get gradient
	deltaw = fn([img, 0, X_label])[0]

	#update
	update = np.sign(deltaw)
	eta = 4
	img += eta*update

	img = (img+mean)[..., ::-1]
	img = np.clip(np.round(img).reshape(224, 224, 3), 0, 255).astype('uint8')

	imsave(os.path.join(ouput_file, index + '.png'), img)
	print('finishing {}'.format(index))