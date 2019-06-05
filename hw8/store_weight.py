import numpy as np
from keras.models import load_model
import sys, os

model = load_model(sys.argv[1])
#model.save_weights(sys.argv[1] + '_weight')

w = model.get_weights()
w = np.array([i.astype('float16') for i in w])
np.save(sys.argv[1] + '_weight.npy', w)
