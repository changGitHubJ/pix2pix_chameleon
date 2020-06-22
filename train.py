from __future__ import print_function, division
import scipy
import random

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from load_data import DataLoader
import numpy as np
import os

import common as c

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        print("Reading...")
        filepath = ["./data/trainImage256.txt",
                    "./data/trainLabel256.txt",
                    "./data/testImage256.txt",
                    "./data/testLabel256.txt"]
        self.data_loader = DataLoader(filepath, img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        print("Building discriminator")
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        print("Building generator...")
        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        # def conv2d(layer_input, filters, f_size=4, bn=True):
        #     """Layers used during downsampling"""
        #     d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        #     d = LeakyReLU(alpha=0.2)(d)
        #     if bn:
        #         d = BatchNormalization(momentum=0.8)(d)
        #     return d

        # def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        #     """Layers used during upsampling"""
        #     u = UpSampling2D(size=2)(layer_input)
        #     u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        #     if dropout_rate:
        #         u = Dropout(dropout_rate)(u)
        #     u = BatchNormalization(momentum=0.8)(u)
        #     u = Concatenate()([u, skip_input])
        #     return u

        # Image input
        # d0 = Input(shape=self.img_shape)

        # # Downsampling
        # d1 = conv2d(d0, self.gf, bn=False)
        # d2 = conv2d(d1, self.gf*2)
        # d3 = conv2d(d2, self.gf*4)
        # d4 = conv2d(d3, self.gf*8)
        # d5 = conv2d(d4, self.gf*8)
        # d6 = conv2d(d5, self.gf*8)
        # d7 = conv2d(d6, self.gf*8)

        # # Upsampling
        # u1 = deconv2d(d7, d6, self.gf*8)
        # u2 = deconv2d(u1, d5, self.gf*8)
        # u3 = deconv2d(u2, d4, self.gf*8)
        # u4 = deconv2d(u3, d3, self.gf*4)
        # u5 = deconv2d(u4, d2, self.gf*2)
        # u6 = deconv2d(u5, d1, self.gf)

        # u7 = UpSampling2D(size=2)(u6)
        # output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        inputs = Input(shape=self.img_shape)

        # encoding ##############
        conv1_1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(inputs)
        conv1_2 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv1_1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

        conv2_1 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(pool1)
        conv2_2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv2_1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

        conv3_1 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(pool2)
        conv3_2 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv3_1)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)

        conv4_1 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(pool3)
        conv4_2 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv4_1)
        #drop4 = Dropout(0.5)(conv4_2)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_2)

        conv5_1 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(pool4)
        conv5_2 = Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv5_1)
        #drop5 = Dropout(0.5)(conv5_2)
        conv_up5 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(UpSampling2D(size=(2, 2))(conv5_2))
        concated5 = concatenate([conv4_2, conv_up5], axis=3)

        # decoding ##############
        conv6_1 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(concated5)
        conv6_2 = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv6_1)
        conv_up6 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(UpSampling2D(size=(2, 2))(conv6_2))
        concated6 = concatenate([conv3_2, conv_up6], axis=3)

        conv7_1 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(concated6)
        conv7_2 = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv7_1)
        conv_up7 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(UpSampling2D(size=(2, 2))(conv7_2))
        concated7 = concatenate([conv2_2, conv_up7], axis=3)

        conv8_1 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(concated7)
        conv8_2 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv8_1)
        conv_up8 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(UpSampling2D(size=(2, 2))(conv8_2))
        concated8 = concatenate([conv1_2, conv_up8], axis=3)

        conv9_1 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(concated8)
        conv9_2 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="zeros")(conv9_1)
        outputs = Conv2D(self.channels, 1, activation="tanh")(conv9_2)

        return Model(input=inputs, output=outputs)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            batchs = self.defineBatchComtents(batch_size)
            for batch_i in range(len(batchs)): #, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                batch_sorted = sorted(batchs[batch_i])
                imgs_A, imgs_B = self.data_loader.load_batch(batch_sorted)
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                string = "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, len(batchs),
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time)
                print (string)
                with open("./training.LOG", "a") as f:
                    f.write(string + "\n")

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/test', exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_test_data(batch_size=3)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/test/%d_%d.png" % (epoch, batch_i))
        plt.close()

    def save(self, output_filename):
        self.generator.save(output_filename[0])
        self.discriminator.save(output_filename[1])
    
    def defineBatchComtents(self, batch_size):
        num = np.linspace(0, c.TRAIN_DATA_SIZE - 1, c.TRAIN_DATA_SIZE)
        num = num.tolist()
        COMPONENT = []
        total_batch = int(c.TRAIN_DATA_SIZE/batch_size)
        for i in range(total_batch):
            component = random.sample(num, batch_size)
            COMPONENT.append(component)
            for j in range(batch_size):
                cnt = 0
                while True:
                    if(num[cnt] == component[j]):
                        num.pop(cnt)
                        break
                    else:
                        cnt += 1
        return COMPONENT

if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=200, batch_size=8, sample_interval=200)
    gan.save(["generator.h5", "discriminator.h5"])