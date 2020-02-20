import numpy as np
import main as m
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer, MaxPooling2D, UpSampling2D, Conv2D
from keras.models import Sequential, Model
import keras

"""
saves encoding of images from dataset to sample principal components from later
"""

IMG_SIZE = 128 
dense_size = 2048
load_path = 'yourpath'

paths = m.initgroup("path to dataset", 30000, 30000)
X = m.load_lfw_dataset(paths, IMG_SIZE)

X = X.astype('float32') / 255.0 - 0.5

X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

encoder = keras.models.load_model('path to location of load_path' + load_path + 'ec.h5')
encoder.name = 'enc'

print(len(X))
print(dense_size)
denseArray = np.zeros((len(X),dense_size))

idx = 0
for img in X:
    denseRep = encoder.predict(img[None])[0]
    denseArray[idx] = np.array(denseRep)
    idx += 1
    
np.save("denseArrHd.npy", denseArray)