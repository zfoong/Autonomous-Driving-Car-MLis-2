# python standard libraries
import os
import re
import random
import fnmatch



import matplotlib.pyplot as plt

# data processing
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
from imgaug import augmenters as img_aug
from random import randint

# tensorflow
import tensorflow as tf
import keras
from keras.models import Sequential  # V2 is tensorflow.keras.xxxx, V1 is keras.xxx
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, AveragePooling2D
from keras.optimizers import Adam, RMSprop
import keras.backend as K
from keras.models import load_model
import Nvidia_Model


def mse_steering(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def mse_speed(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def zoom(image):
    zoom = img_aug.Affine(scale=(1, 1.3))  # zoom from 100% (no zoom) to 130%
    image = zoom.augment_image(image)
    return image

def pan(image):
    # pan left / right / up / down about 10%
    pan = img_aug.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image


def adjust_brightness(image):
    # increase or decrease brightness by 30%
    brightness = img_aug.Multiply((0.7, 1.3))
    image = brightness.augment_image(image)
    return image


def blur(image):
    kernel_size = random.randint(1, 5)  # kernel larger than 5 would make the image way too blurry
    image = cv2.blur(image, (kernel_size, kernel_size))

    return image


def random_flip(image, steering_angle):
    is_flip = random.randint(0, 1)
    if is_flip == 1:
        # randomly flip horizon
        image = cv2.flip(image, 1)
        steering_angle = 180 - steering_angle

    return image, steering_angle


def random_augment(image, steering_angle):
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = blur(image)
    if np.random.rand() < 0.5:
        image = adjust_brightness(image)

    image, steering_angle = random_flip(image, steering_angle)

    return image, steering_angle

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height / 2):, :, :]  # remove top third of the image, as it is not relevant for lane following
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3, 3), 0)  # Blurs image
    image = cv2.resize(image, (200, 66))  # input image size (200,66) Nvidia model
    image = image / 255  # normalizing
    return image

def image_data_generator(image_paths, steering_angle, batch_size, is_training):
    while True:
        batch_images = []
        batch_speeds = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image_path = image_paths[random_index]
            image = cv2.imread(image_paths[random_index])
            angle = steering_angle[random_index]
            if is_training:
                #training: augment image
                image, steering_angle = random_augment(image, steering_angle)

            image = img_preprocess(image)
            batch_images.append(image)
            batch_speeds.append(angle)

        yield (np.asarray(batch_images), np.array(batch_speeds))

def nvidia_model():
    model = Sequential(name='Nvidia_Model')

    # elu=Expenential Linear Unit, similar to leaky Relu
    # skipping 1st hiddel layer (nomralization layer), as we have normalized the data

    # Convolution Layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu', data_format="channels_last", name='cov_1'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu', name='cov_2'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu', name='cov_3'))
    model.add(Conv2D(64, (3, 3), activation='elu', name='cov_4'))
    model.add(Dropout(0.2))  # not in original model. added for more robustness
    model.add(Conv2D(64, (3, 3), activation='elu', name='cov_5'))
    # model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dropout(0.5))  # not in original model. added for more robustness
    model.add(Dense(100, activation='elu', name='dense_1'))
    model.add(Dense(50, activation='elu', name='dense_2'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='elu', name='dense_3'))

    # output layer: turn angle (from 45-135, 90 is straight, <90 turn left, >90 turn right)
    model.add(Dense(2, activation='linear', name='output'))

    # since this is a regression problem not classification problem,
    # we use MSE (Mean Squared Error) as loss function
    optimizer = Adam(lr=1e-3)  # lr is learning rate
    return model

def main():
    data_dir = 'oval'
    file_list = os.listdir(data_dir)
    image_paths = []
    steering_angles = []
    car_speed = []
    angles_speed = []
    pattern = "*.png"

    for filename in file_list:
        if fnmatch.fnmatch(filename, pattern):
            image_paths.append(os.path.join(data_dir, filename))
            filename_split = re.split('[_.]', filename)
            angle = int(filename_split[1])    # 092 part of video01_143_092.png is the angle. 90 is go straight
            speed = int(filename_split[2])
            steering_angles.append(angle)
            car_speed.append(speed)
            angles_speed.append([angle, speed])

    X_train, X_valid, y_train, y_valid = train_test_split(image_paths, angles_speed, test_size=0.2, shuffle= True)
    print("Training data: %d\nValidation data: %d" % (len(X_train), len(X_valid)))

    y_train = np.transpose(np.array(y_train))
    y_valid = np.transpose(np.array(y_valid))

    #model = nvidia_model()
    model = load_model('lane_navigation_check_CombinedDataSet(max_pool_FF).h5', custom_objects={'mse_steering': mse_steering, 'mse_speed': mse_speed})
    model.layers[0].trainable = False
    model.layers[1].trainable = False
    model.layers[2].trainable = False
    model.layers[3].trainable = False
    model.layers[4].trainable = False
    model.layers[5].trainable = False
    model.layers[6].trainable = False
    model.layers[7].trainable = False
    model.layers[8].trainable = False
    model.layers[9].trainable = False
    model.layers[10].trainable = False
    model.layers[11].trainable = False

    # since this is a regression problem not classification problem,
    # we use MSE (Mean Squared Error) as loss function
    optimizer = Adam(lr=1e-3)  # lr is learning rate
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', mse_steering, mse_speed])

    print(model.summary())

    # saves the model weights after each epoch if the validation loss decreased
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('lane_navigation_check_TL_fine_tune.h5', monitor='val_loss',
                                                            verbose=1, save_best_only=True)

    history = model.fit_generator(Nvidia_Model.image_data_generator(X_train, y_train, batch_size=100, is_training=True),
                                  steps_per_epoch=300,
                                  epochs=5,
                                  validation_data=Nvidia_Model.image_data_generator(X_valid, y_valid, batch_size=len(X_valid),
                                                                       is_training=False),
                                  validation_steps=1,
                                  verbose=1,
                                  shuffle=1,
                                  callbacks=[checkpoint_callback])

    # always save model output as soon as model finishes training
    model.save('lane_navigation_final_TL_fine_tune.h5')

    plt.plot(history.history['loss'], color='blue')
    plt.plot(history.history['val_loss'], color='red')
    plt.legend(["training loss", "validation loss"])
    plt.show()

    plt.plot(history.history['mse_steering'], color='blue')
    plt.plot(history.history['mse_speed'], color='red')
    plt.legend(["steering angle loss", "speed loss"])
    plt.show()

if __name__ == '__main__':
    main()
