import numpy as np
from keras.models import Sequential, load_model
import sys, os
from sklearn.cluster import KMeans
import pickle
from skimage.io import imread
from MulticoreTSNE import MulticoreTSNE as TSNE

input_file = sys.argv[1]
# function of loading data 
data = []
for i in range(1, 40001):
	index = str(i).zfill(6)
	img = imread(os.path.join(input_file, index + '.jpg'))
	data.append(img)
	if i % 1000 == 999:
		print('finishing loding',i+1)
X = np.asarray(data).astype('float')
print('finish loading data')
X /= 255

# transfer data to vector
encoder = load_model(sys.argv[2])
vec = encoder.predict(X)
vec = vec.reshape(40000, -1)
print('vec_shape =',vec.shape)
#np.save('test_vec.npy', vec)

tsne = TSNE(n_jobs = 20)
Y = tsne.fit_transform(vec)
print('finish TSNE')

km = KMeans(2, random_state = 66).fit(Y)
print('finish 2_means')

test_data = np.genfromtxt(sys.argv[3], delimiter = ',').astype('int32')
test_data = np.delete(test_data, 0, 0)
test_data = np.delete(test_data, 0, 1)
test_data -= 1

label = km.labels_

f = open(sys.argv[4], 'w')
wt_str = 'id,label\n'
f.write(wt_str)
for i in range(test_data.shape[0]):
	name_0 = test_data[i, 0]
	name_1 = test_data[i, 1]

	# output
	opt = 0
	if label[name_1] == label[name_0]:
		opt = 1
	wt_str = str(i)+','+str(opt)+'\n'
	f.write(wt_str)

	# check 
	if i % 1000 == 999:
		print(i + 1)
