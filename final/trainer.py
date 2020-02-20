from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer, MaxPooling2D, UpSampling2D, Conv2D
from keras.models import Sequential, Model
import keras
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import main as m

"""
Train the neural network, save to given path.
"""

IMG_SIZE = 128    ##size of image (all square)
set_start = 30000 ##starting idx of dataset
set_size = 30000 ##size of sample from dataset

load = True        ##boolean: load or init. new neural network
load_path = 'hd2048' 

dense_size = 4096 ##dense representation in autoencoder
epoch = 6
rep = 300 ##how many times repeat training (can be very large and just terminate manually when done)
save_path = 'hd2048'
show = 0 ##how often draw results of autoencoder (zero if never show results during training)

check = False ##show images from outside dataset (upload custom images)

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))

def build_autoencoder(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(Flatten(input_shape= img_shape)) 
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(Dense(np.prod(img_shape), input_shape=(code_size,)))
    decoder.add(Reshape(img_shape))
    return encoder, decoder

##draws original image + result from encoding
def visualize(img,encoder,decoder):\
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()

##draws original image + result from encoding
def vis2(img, autoenc):
    reco = autoenc.predict(img[None])[0]
    
    plt.subplot(1,3,1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()


paths = m.initgroup("path to dataset here", set_size, set_start)
X = m.load_lfw_dataset(paths, IMG_SIZE)

X = X.astype('float32') / 255.0 - 0.5

IMG_SHAPE = X.shape[1:]


X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)



if load:
    encoder = keras.models.load_model('path containing load file here' + load_path + 'ec.h5')
    encoder.name = 'enc'
    decoder = keras.models.load_model('path containing load file here' + load_path + 'dc.h5')
    decoder.name = 'dec'
else:
    encoder, decoder = build_autoencoder(IMG_SHAPE, dense_size)

inp = Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)
autoencoder = Model(inp,reconstruction)
autoencoder.compile(optimizer='adam', loss='mse')

ct = 1
for x in range(rep):
    
    print("Trial " + str(x))
    history = autoencoder.fit(x=X_train, y=X_train, batch_size=2000, epochs=epoch,
                validation_data=[X_test, X_test])
    encoder.save('path containing load file here' + save_path +'ec.h5')
    decoder.save('path containing load file here' + save_path +'dc.h5')
    autoencoder.save('path containing load file here' + save_path +'ac.h5')
    ct += 1
    if show != 0 and ct % show == 0:
        for i in range(20):
            img = X_test[i]
            visualize(img,encoder,decoder)


if check:
    x_test = m.load_lfw_dataset(m.initgroup('path containing test image dataset here'))
    x_test = x_test.astype('float32') / 255.0 - 0.5
    for x in x_test:
        visualize(x,encoder,decoder)

for i in range(20):
    img = X_test[i]
    visualize(img,encoder,decoder)

