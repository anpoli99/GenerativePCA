from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
import tensorflow as tf

##generalized method holder

##get file names from folder
def initgroup(pathname, img_ct = 12500, img_start= 0):
    files = [f for f in listdir(pathname) if isfile(join(pathname, f))]
    rtnr = []
    ct = 0
    for f in files:
        ct += 1
        if(ct < img_start):
            continue
        if ct - img_start > img_ct:
            break
        try:
            rtnr.append(pathname + "\\" + f)
        except:
            print("Failed to load \"" + pathname + "\\" + f + "\"" )
    return rtnr

##read image from path
def decode_image(path, IMG_SIZE = 64):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    return img


def load_dataset(paths, IMG_SIZE = 64):

    all_photos = []
    ct = 0
    for path in paths:
        # Prepare image
        ct += 1
        if(ct % 1000 == 0):
            print(ct)
        img = decode_image(path, IMG_SIZE)
        all_photos.append(img)

    all_photos = np.stack(all_photos).astype('uint8')
    return all_photos
