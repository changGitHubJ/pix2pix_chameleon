import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from matplotlib import cm

import common as c
import load_data as data

def readImages(filename, size):
    img = np.zeros((c.IMG_SIZE, c.IMG_SIZE, 3), dtype=np.int32)
    imgs = []
    fileImg = open(filename)
    for k in range(size):
        line = fileImg.readline()
        if(not line):
            break
        val = line.split(',')
        for i in range(c.IMG_SIZE):
            for j in range(c.IMG_SIZE):
                img[i, j, 0] = float(val[3*(256*i + j) + 0 + 1])
                img[i, j, 1] = float(val[3*(256*i + j) + 1 + 1])
                img[i, j, 2] = float(val[3*(256*i + j) + 2 + 1])
        imgs.append(img.copy())
    imgs = np.array(imgs)/127.5 - 1.
    return imgs

if __name__=='__main__':    
    with tf.device("/gpu:0"):
        filepath = ["./data/testImage256.txt",
                    "./data/testLabel256.txt"]
        test_image = readImages(filepath[0], c.TEST_DATA_SIZE)
        test_label = readImages(filepath[1], c.TEST_DATA_SIZE)
        model = keras.models.load_model("generator.h5")
        fake = model.predict(test_image)   # OK
        for i in range(c.TEST_DATA_SIZE):
            image256 = np.zeros([c.IMG_SIZE, c.IMG_SIZE, 3], dtype=np.int32)
            fake256 = np.zeros([c.IMG_SIZE, c.IMG_SIZE, 3], dtype=np.int32)
            label256 = np.zeros([c.IMG_SIZE, c.IMG_SIZE, 3], dtype=np.int32)
            for m in range(c.IMG_SIZE):
                for n in range(c.IMG_SIZE):
                    image256[m, n, 0] = int((test_image[i][m, n, 0] + 1)*127.5)
                    image256[m, n, 1] = int((test_image[i][m, n, 1] + 1)*127.5)
                    image256[m, n, 2] = int((test_image[i][m, n, 2] + 1)*127.5)
                    fake256[m, n, 0] = int((fake[i][m, n, 0] + 1)*127.5)
                    fake256[m, n, 1] = int((fake[i][m, n, 1] + 1)*127.5)
                    fake256[m, n, 2] = int((fake[i][m, n, 2] + 1)*127.5)
                    label256[m, n, 0] = int((test_label[i][m, n, 0] + 1)*127.5)
                    label256[m, n, 1] = int((test_label[i][m, n, 1] + 1)*127.5)
                    label256[m, n, 2] = int((test_label[i][m, n, 2] + 1)*127.5)
            plt.figure(figsize=[15, 4])
            plt.subplot(1, 3, 1)
            fig = plt.imshow(image256)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.subplot(1, 3, 2)
            fig = plt.imshow(fake256)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)  
            plt.subplot(1, 3, 3)
            fig = plt.imshow(label256)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False) 
            plt.show()