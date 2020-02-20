from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer, MaxPooling2D, UpSampling2D, Conv2D
from keras.models import Sequential, Model
import keras
import main as m


"""
visualize results of autoencoder, tests encoding/decoding 
"""

load_path = 'modern1024' ##path to load neural network
check = True  ##draw results from images outside dataset (upload custom test images)
IMG_SIZE = 64 ##size of images
set_size = 1000 ##size of dataset for testing
set_start = 30000 ##index of starting point

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))

def visualize(img,encoder,decoder):
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()

def vis2(img, autoenc):
    reco = autoenc.predict(img[None])[0]
    
    plt.subplot(1,3,1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()


encoder = keras.models.load_model('path containing load_path' + load_path + 'ec.h5')
encoder.name = 'enc'
decoder = keras.models.load_model('path containing load_path' + load_path + 'dc.h5')
decoder.name = 'dec'

#for testing images outside dataset
x_test = m.load_lfw_dataset(m.initgroup('your path here (to folder w/ test images) '))
x_test = x_test.astype('float32') / 255.0 - 0.5

for x in x_test:
    if not check:
        break
    visualize(x,encoder,decoder)


paths = m.initgroup("path to dataset here", set_size, set_start)
X = m.load_lfw_dataset(paths, IMG_SIZE)

X = X.astype('float32') / 255.0 - 0.5 #normalize

for x in X:
    visualize(x, encoder, decoder)