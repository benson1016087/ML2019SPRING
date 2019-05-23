import os
import sys
import numpy as np 
from skimage.io import imread, imsave

IMAGE_PATH = sys.argv[1]

# Images for compression & reconstruction
test_image = sys.argv[2]

# Number of principal components used
k = 5

def process(M): 
	M -= np.min(M)
	M /= np.max(M)
	M = (M * 255).astype('uint8')
	return M

filelist = os.listdir(IMAGE_PATH) 
# Record the shape of images
img_shape = imread(os.path.join(IMAGE_PATH, filelist[0])).shape 

img_data = []
for filename in filelist:
    tmp = imread(os.path.join(IMAGE_PATH,filename))  
    img_data.append(tmp.flatten())

training_data = np.array(img_data).astype('float32')

# Calculate mean & Normalize
mean = np.mean(training_data, axis = 0)  
training_data -= mean 

# Use SVD to find the eigenvectors 

u, s, v = np.linalg.svd(training_data, full_matrices = False)  
'''
np.save('u.npy', u)
np.save('s.npy', s)
np.save('v.npy', v)

u = np.load('u.npy')
s = np.load('s.npy')
v = np.load('v.npy')
'''
print('end solving u, s, v')


# Load image & Normalize
picked_img = imread(os.path.join(IMAGE_PATH,test_image))  
X = picked_img.flatten().astype('float32') 
X -= mean

# Reconstruction
reconstruct = process(v[:k].T.dot(v[:k].dot(X)) + mean)
imsave(sys.argv[3], reconstruct.reshape(img_shape))
'''
# prob_a
average = process(mean)
imsave('average.jpg', average.reshape(img_shape))  

# prob_b
for x in range(5):
	eigenface = process(v[x])
	imsave(str(x) + '_eigenface.jpg', eigenface.reshape(img_shape))

# prob_d
for i in range(5):
	number = s[i] * 100 / sum(s)
	print(number)
'''
