import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import random

import common as c

class DataLoader():
    def __init__(self, filepath, img_res=(128, 128)):
        self.img_res = img_res
        self.train_image = self.readImages(filepath[0], c.TRAIN_DATA_SIZE)
        self.train_label = self.readImages(filepath[1], c.TRAIN_DATA_SIZE)
        self.test_image = self.readImages(filepath[2], c.TEST_DATA_SIZE)
        self.test_label = self.readImages(filepath[3], c.TEST_DATA_SIZE)

    def load_test_data(self, batch_size):
        num = np.linspace(0, c.TEST_DATA_SIZE - 1, c.TEST_DATA_SIZE)
        batch_images = np.random.choice(num, size=batch_size)
        imgs_A = []
        imgs_B = []
        for i in batch_images:
            img_A = self.test_label[int(i), :].reshape([self.img_res[0], self.img_res[1], 3])
            img_B = self.test_image[int(i), :].reshape([self.img_res[0], self.img_res[1], 3])

            # h, w, _ = img.shape
            # _w = int(w/2)
            # img_A, img_B = img[:, :_w, :], img[:, _w:, :]

            # img_A = scipy.misc.imresize(img_A, self.img_res)
            # img_B = scipy.misc.imresize(img_B, self.img_res)

            # # If training => do random flip
            # if not is_testing and np.random.random() < 0.5:
            #     img_A = np.fliplr(img_A)
            #     img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch):
        imgs_A, imgs_B = [], []
        for j in batch:
            img_A = self.train_label[int(j), :].reshape([self.img_res[0], self.img_res[1], 3])
            img_B = self.train_image[int(j), :].reshape([self.img_res[0], self.img_res[1], 3])
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.
        return imgs_A, imgs_B
    
    def readImages(self, filename, size):
        images = np.zeros((size, self.img_res[0], self.img_res[1], 3), dtype=np.int32)
        fileImg = open(filename)
        for k in range(size):
            line = fileImg.readline()
            if(not line):
                break
            val = line.split(',')
            for i in range(self.img_res[0]):
                for j in range(self.img_res[1]):
                    images[k, i, j, 0] = float(val[3*(256*i + j) + 0 + 1])
                    images[k, i, j, 1] = float(val[3*(256*i + j) + 1 + 1])
                    images[k, i, j, 2] = float(val[3*(256*i + j) + 2 + 1])
        return images
