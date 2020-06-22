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
                # add blot
                blot_arr = np.array(resized)
                c1 = random.uniform(50, 205) # center
                c2 = random.uniform(50, 205) # center
                a = random.uniform(5, 20)
                b = random.uniform(5, 20)
                theta = random.uniform(0, 180)
                t = 0.0
                while True:
                    v1 = a*math.cos(t + theta*math.pi/180.0)
                    v2 = b*math.sin(t + theta*math.pi/180.0)
                    norm = math.sqrt(v1*v1 + v2*v2)
                    v1 /= norm
                    v2 /= norm
                    d = 0.0
                    while True:
                        x = c1 + v1*d
                        y = c2 + v2*d
                        x = int(x)
                        y = int(y)
                        if(x < 0): x = 0
                        if(x > 255): x = 255
                        if(y < 0): y = 0
                        if(y > 255): y = 255
                        blot_arr[int(x), int(y), 0] = 0
                        blot_arr[int(x), int(y), 1] = 0
                        blot_arr[int(x), int(y), 2] = 0
                        d += 0.5
                        r = math.sqrt(a*math.cos(t + theta)*a*math.cos(t + theta) + b*math.sin(t + theta)*b*math.sin(t + theta))
                        if(d > r):
                            break
                    t += math.pi*0.01
                    if(t > math.pi*2.0):
                        break
                blot_img = Image.fromarray(blot_arr)
                # plt.imshow(blot_img)
                # plt.show()
                blot_img.save(directory + '/' + file + '_256_blot.jpg')
            except:
                print("cannot output: " + directory + '/' + file + '_256.jpg')