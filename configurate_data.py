import math
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image

import common as c

directory = ['./images/lizard/']

if __name__ == "__main__":

    loop = 0
    counter = 0
    train_img = []
    train_lbl = []
    test_img = []
    test_lbl = []
    while True:
        fname_img = directory[0] + str(loop) + "_256_blot.jpg"
        fname_lbl = directory[0] + str(loop) + "_256.jpg"
        try:
            img = Image.open(fname_img)
            lbl = Image.open(fname_lbl)

            print(fname_img + ", " + fname_lbl)
            img_arr = np.array(img)
            lbl_arr = np.array(lbl)

            if(counter < c.TRAIN_DATA_SIZE):
                line_img = str(counter)
                line_lbl = str(counter)
                for i in range(c.IMG_SIZE):
                    for j in range(c.IMG_SIZE):
                        line_img = line_img + ',' + str(img_arr[i, j, 0]) + ',' + str(img_arr[i, j, 1]) + ',' + str(img_arr[i, j, 2])
                        line_lbl = line_lbl + ',' + str(lbl_arr[i, j, 0]) + ',' + str(lbl_arr[i, j, 1]) + ',' + str(lbl_arr[i, j, 2])
                line_img = line_img + '\n'
                line_lbl = line_lbl + "\n"
                train_img.append(line_img)
                train_lbl.append(line_lbl)
                counter += 1
            else:
                line_img = str(counter - c.TRAIN_DATA_SIZE)
                line_lbl = str(counter - c.TRAIN_DATA_SIZE)
                for i in range(c.IMG_SIZE):
                    for j in range(c.IMG_SIZE):
                        line_img = line_img + ',' + str(img_arr[i, j, 0]) + ',' + str(img_arr[i, j, 1]) + ',' + str(img_arr[i, j, 2])
                        line_lbl = line_lbl + ',' + str(lbl_arr[i, j, 0]) + ',' + str(lbl_arr[i, j, 1]) + ',' + str(lbl_arr[i, j, 2])
                line_img = line_img + '\n'
                line_lbl = line_lbl + "\n"
                test_img.append(line_img)
                test_lbl.append(line_lbl)
                if(counter > c.TRAIN_DATA_SIZE + c.TEST_DATA_SIZE):
                    break
                else:
                    counter += 1
        except:
            print("cannot open" + fname_img)
        finally:
            loop += 1

    if not os.path.exists('./data'):
        os.mkdir('./data')
    with open('./data/trainImage256.txt', 'w') as f:
        for line in train_img:
            f.write(line)
    with open('./data/trainLabel256.txt', 'w') as f:
        for line in train_lbl:
            f.write(line)
    with open('./data/testImage256.txt', 'w') as f:
        for line in test_img:
            f.write(line)
    with open('./data/testLabel256.txt', 'w') as f:
        for line in test_lbl:
            f.write(line)
 