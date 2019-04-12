import sys, csv, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import plot_model
from termcolor import colored
from termcolor import cprint
import keras.backend as K
from lime import lime_image
from skimage.segmentation import slic

#extract data
data = pd.read_csv(sys.argv[1])
Y = data['label'].values.reshape(-1, 1)
data_X = data['feature'].str.split(' ', expand = True)
X = data_X.values.astype('float')

#generate X and Y
X = (X / 255).reshape(-1, 48, 48, 1)
Y = np.hstack((Y == 0, Y == 1, Y == 2, Y == 3, Y == 4, Y == 5, Y == 6))
Y = np.argmax(Y, axis = 1)
model = load_model(sys.argv[2])
labels = [0, 416, 21, 7, 3, 15, 4]
print('finish loading')

#lime
np.random.seed(0)
X_rgb = np.stack((X, X, X), axis = 3)

# two functions that lime image explainer requires
def pred(data):
    return model.predict(np.mean(data.reshape(-1, 48, 48, 3), axis = 3).reshape(-1, 48, 48, 1)).reshape(-1, 7)

def seg(data):
    return slic(data, n_segments = 100)

for idx in labels:
    # Initiate explainer instance
    explainer = lime_image.LimeImageExplainer()
    predict_result = pred(X_rgb[idx])
    predict_label = predict_result.argmax()
    print("ID: {}, Prediction: {}".format(idx, predict_label))
    # Get the explaination of an image
    explaination = explainer.explain_instance(
                                image=X_rgb[idx].reshape(48, 48, 3), 
                                classifier_fn=pred,
                                segmentation_fn=seg,
                                random_seed = 2866
                            )

    # Get processed image
    image, mask = explaination.get_image_and_mask(
                                    label=Y[idx],
                                    positive_only=False,
                                    hide_rest=False,
                                    num_features=5,
                                    min_weight=0.0
                                )

    # save the image
    plt.imsave(os.path.join(sys.argv[3], "fig3_" + str(predict_label) + ".jpg"), image)
print('finish lime')


#saliency
np.random.seed(0)
inputImage = model.input

for idx in labels:
    pred_prob = model.predict(X[idx].reshape(-1, 48, 48, 1))
    pred_label = pred_prob.argmax()
    #print(model.output.shape, pred_label)
    tensorTarget = model.output[:, pred_label]
    tensorGradients = K.gradients(tensorTarget, inputImage)[0]
    fn = K.function([inputImage, K.learning_phase()], [tensorGradients])

    # heatmap processing
    arrayGradients = fn([X[idx].reshape(-1, 48, 48, 1), 0])[0].reshape(48, 48, -1)
    arrayGradients = np.max(np.abs(arrayGradients), axis=-1, keepdims=True)

    # normalize center on 0., ensure std is 0.1
    arrayGradients = (arrayGradients - np.mean(arrayGradients)) / (np.std(arrayGradients) + 1e-5)
    arrayGradients *= 0.1

    # clip to [0, 1]
    arrayGradients += 0.5
    arrayGradients = np.clip(arrayGradients, 0, 1)

    arrayHeatMap = arrayGradients.reshape(48, 48)
    ### End heatmap processing ###

    print("ID: {}, Prediction: {}".format(idx, pred_label))

    # show original image
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(1, 3, 1)
    axx = ax.imshow((X[idx]*255).reshape(48, 48), cmap="gray")
    plt.tight_layout()

    # show Heat Map
    ax = fig.add_subplot(1, 3, 2)
    axx = ax.imshow(arrayHeatMap, cmap=plt.cm.jet)
    plt.colorbar(axx)
    plt.tight_layout()

    # show Saliency Map
    floatThreshold = 0.55
    arraySee = (X[idx]*255).reshape(48, 48)
    arraySee[np.where(arrayHeatMap <= floatThreshold)] = np.mean(arraySee)

    ax = fig.add_subplot(1, 3, 3)
    axx = ax.imshow(arraySee, cmap="gray")
    plt.colorbar(axx)
    plt.tight_layout()
    fig.suptitle("Class {}".format(pred_label))
    plt.savefig(os.path.join(sys.argv[3], "fig1_" + str(pred_label) + ".jpg"))
print('finish saliency map')


#filter
X_21 = X[21]

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

input_image = model.input

layer_dict = dict([layer.name, layer] for layer in model.layers)
listLayerNames = [layer for layer in layer_dict.keys() if "activation" in layer or "conv2d" in layer][:8]
# define the function that input is an image and calculate the image through each layer until the output layer that we choose
listCollectLayers = [K.function([input_image, K.learning_phase()], [layer_dict[name].output]) for name in listLayerNames] 
intFilters = [32, 32, 64, 64, 128, 128]

for cnt, fn in enumerate(listCollectLayers):
    if cnt < 2: 
        continue
    #dealing with output
    arrayPhoto = X_21.reshape(1, 48, 48, 1)
    listLayerImage = fn([arrayPhoto, 0])
    fig = plt.figure(figsize=(16, 16))

    for i in range(intFilters[cnt]):
        ax = fig.add_subplot(intFilters[cnt] // 8, 8, i+1)
        ax.imshow(listLayerImage[0][0, :, :, i], cmap="Blues")
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.xlabel("filter {}".format(i))
        plt.tight_layout()
    fig.suptitle("Output of {} (Given image{})".format(listLayerNames[cnt], 21))
    plt.savefig(os.path.join(sys.argv[3], "fig2_2.jpg"))
    print('saving output done')


    # dealing with filter
    eta = 3e-4
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    fig2 = plt.figure(figsize=(16, 17))
    for i in range(intFilters[cnt]):
        layer_output = layer_dict[listLayerNames[cnt]].output
        loss = K.mean(layer_output[:, :, :, i])
        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_image)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iterate = K.function([input_image], [loss, grads])

        # start from a gray image with some noise
        np.random.seed(66)
        input_img_data = (np.random.random((1, 48, 48, 1)) * 20 + 128) / 255
        m = np.zeros((1, 48, 48, 1))
        v = np.zeros((1, 48, 48, 1))
        # run gradient ascent for 20 steps
        for time in range(100):
            loss_value, deltaw = iterate([input_img_data])
            m = beta1 * m + (1 - beta1) * deltaw
            v = beta2 * v + (1 - beta2) * (deltaw ** 2)
            m_bar = m / (1 - beta1 ** (time+1))
            v_bar = v / (1 - beta2 ** (time+1))
            input_img_data =  input_img_data + eta * m_bar / (np.sqrt(v_bar) + epsilon)
        img = input_img_data[0]
        img = deprocess_image(img)

        ax = fig2.add_subplot(intFilters[cnt] // 8, 8, i+1)
        ax.imshow(img.reshape(48, 48), cmap="winter")
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.xlabel("filter {}".format(i))
        plt.tight_layout()
    fig2.suptitle("Filter {}:".format(listLayerNames[cnt], 21))
    plt.savefig(os.path.join(sys.argv[3], "fig2_1.png"))

    print('saving filter done')
    exit()

#resource: https://github.com/machineCYC/2017MLSpring_Hung-yi-Lee/blob/master/HW3/Plot.py
#resource: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html?fbclid=IwAR1mmKdsxQzDWbUBXwiZ88lwVKXnPcdRaIqtYRAOGqlx2OakOsvFnToapbk