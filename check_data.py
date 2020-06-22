import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import random

from PIL import Image

NUM = 30
IMG_SIZE = 256
OUTPUT_SIZE = 256*256

def readImages(filename):
    images = np.zeros((NUM, IMG_SIZE, IMG_SIZE, 3), dtype=np.int32)
    fileImg = open(filename)
    for k in range(NUM):
        line = fileImg.readline()
        if(not line):
            break
        val = line.split(',')
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                images[k, i, j, 0] = float(val[3*(256*i + j) + 0 + 1])
                images[k, i, j, 1] = float(val[3*(256*i + j) + 1 + 1])
                images[k, i, j, 2] = float(val[3*(256*i + j) + 2 + 1])
    return images

if __name__=='__main__':
    train_image = readImages('./data/testImage256.txt')
    train_label = readImages('./data/testLabel256.txt')

    for i in range(NUM):
        plt.figure(figsize=[10, 4])
        plt.subplot(1, 2, 1)
        fig = plt.imshow(train_image[i, :].reshape([IMG_SIZE, IMG_SIZE, 3]))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)    
        
        plt.subplot(1, 2, 2)
        fig = plt.imshow(train_label[i, :].reshape([IMG_SIZE, IMG_SIZE, 3]))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.show()
