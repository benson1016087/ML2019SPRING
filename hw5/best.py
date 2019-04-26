import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
import sys, os
from skimage.io import imread, imsave
from PIL import Image

input_file = sys.argv[1]
ouput_file = sys.argv[2]
model = models.resnet50(pretrained=True)
model.eval()

is_cuda = torch.cuda.is_available()
if is_cuda:
	print("Using GPU")
	model = model.cuda()
else:
	print("Using CPU")
    
pre_mean = [0.485, 0.456, 0.406]
pre_std = [0.229, 0.224, 0.225]
post_mean = np.array([-0.485, -0.456, -0.406])
post_std = np.array([1/0.229, 1/0.224, 1/0.225])
add_std = [0.229/255, 0.224/255, 0.225/255]

preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = pre_mean, std = pre_std)])
preprocess_add = transforms.Compose([transforms.Normalize(mean = [0, 0, 0], std = add_std)])
def de_preprocess(img_np):
	img_np = img_np.reshape(3, 224, 224)
	img_np = np.transpose(img_np, axes = [1, 2, 0])
	img_np /= post_std
	img_np -= post_mean
	img_np = np.round(np.clip(img_np, 0, 1)*255).astype('uint8')
	#print(img_np)
	return img_np

criterion = nn.CrossEntropyLoss()
epsilon = 4

for t in range(200):
	index = str(t).zfill(3)
	img = Image.open(os.path.join(input_file, index + '.png'))
	img_torch = preprocess(img).cuda()
	img_torch = img_torch.unsqueeze(0)
	img_torch.requires_grad = True
	zero_gradients(img_torch)

	output = model(img_torch)
	target_label = torch.zeros((1, ))
	target_label[0] = output.argmax().cuda()
	loss = criterion(output, target_label.long().cuda())
	loss.backward() 

	add_sign = img_torch.grad.sign_()
	for i in range(100):
		img_torch = img_torch + 1 * add_sign / 255 / 0.224
		# output
		img_np = img_torch.cpu().detach().numpy()
		img_output = de_preprocess(img_np)

		#test
		img_test = np.copy(img_output)
		img_test = preprocess(img_test).cuda()
		img_test = img_test.unsqueeze(0)
		img_test.requires_grad = True
		zero_gradients(img_test)

		output = model(img_test)
		test_label = output.argmax().cuda()
		if int(target_label[0].cpu().detach().numpy()) != int(test_label.cpu().detach().numpy()):
			break
		print('i =', i)

	
	# saving figure
	imsave(os.path.join(ouput_file, index + '.png'), img_output)
	print('finishing {}'.format(index))
	#exit()

# Soruce: https://hackmd.io/s/H169qqFKE