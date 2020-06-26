import math
import numpy as np
import os
import random

from matplotlib import pyplot as plt
from PIL import Image

if __name__ == "__main__":
    directories = ['lizard']
    for directory in directories:
        files = os.listdir(directory)
        for file in files:
            try:
                img = Image.open(directory + '/' + file)
                # plt.imshow(img)
                # plt.show()
                width, height = img.size
                if(height >= width):
                    resized = Image.new(img.mode, (height, height), (255, 255, 255))
                    resized.paste(img, ((height - width)//2, 0))
                else:
                    resized = Image.new(img.mode, (width, width), (255, 255, 255))
                    resized.paste(img, (0, (width - height)//2))
                resized = resized.resize((256, 256), Image.LANCZOS)
                # plt.imshow(resized)
                # plt.show()
                file = file.split('.')[0]
                resized.save(directory + '/' + file + '_256.jpg')
                # blur
                blured = Image.new(img.mode, (256, 256), (255, 255, 255))
                blured.paste(resized, (0, 0))
                size = random.randint(32, 128)
                blured = blured.resize((size, size), Image.LANCZOS)
                blured = blured.resize((256, 256), Image.LANCZOS)
                # plt.imshow(blured)
                # plt.show(blured)
                blured.save(directory + '/' + file + '_256_blot.jpg')
            except:
                print("cannot output: " + directory + '/' + file + '_256.jpg')