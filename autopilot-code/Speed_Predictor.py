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
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, AveragePooling2D, ZeroPadding2D
from keras.optimizers import Adam, RMSprop
import keras.backend as K
from keras.models import load_model


def my_imread(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

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


def random_augment(image):
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = blur(image)
    if np.random.rand() < 0.5:
        image = adjust_brightness(image)

    return image

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height / 3):, :, :]  # remove top third of the image, as it is not relevant for lane following
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3, 3), 0)  # Blurs image
    # image_grey = np.uint8(image)
    # image_edge = cv2.Canny(image_grey, 200, 400)  # detects line edges (lanes)
    # image_edge = cv2.cvtColor(image_edge, cv2.COLOR_GRAY2BGR)
    # image = cv2.addWeighted(image, 1, image_edge, 1, 0)
    image = cv2.resize(image, (200, 66))  # input image size (200,66) Nvidia model
    image = image / 255  # normalizing
    return image

def nvidia_model():
    model = Sequential(name='Alex_Net_modified')

    # elu=Expenential Linear Unit, similar to leaky Relu
    # skipping 1st hiddel layer (nomralization layer), as we have normalized the data

    # Convolution Layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu', data_format="channels_last"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.2))  # not in original model. added for more robustness
    model.add(Conv2D(64, (3, 3), activation='elu'))
    # model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dropout(0.2))  # not in original model. added for more robustness
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))

    # output layer: turn angle (from 45-135, 90 is straight, <90 turn left, >90 turn right)
    model.add(Dense(1, activation='sigmoid'))

    # since this is a regression problem not classification problem,
    # we use MSE (Mean Squared Error) as loss function
    optimizer = Adam(lr=1e-3)  # lr is learning rate
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['mae', 'mse', 'accuracy'])

    return model



def image_data_generator(image_paths, car_speed, batch_size, is_training):
    while True:
        batch_images = []
        batch_speeds = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image_path = image_paths[random_index]
            image = cv2.imread(image_paths[random_index])
            speed = car_speed[random_index]
            if is_training:
                #training: augment image
                image = random_augment(image)

            image = img_preprocess(image)
            batch_images.append(image)
            batch_speeds.append(speed)

        yield (np.asarray(batch_images), np.array(batch_speeds))

def main():
    print(f'tf.__version__: {tf.__version__}')
    print(f'keras.__version__: {keras.__version__}')
    np.set_printoptions(formatter={'float_kind': lambda x: "%.4f" % x})
    pd.set_option('display.width', 300)
    pd.set_option('display.float_format', '{:,.4f}'.format)
    pd.set_option('display.max_colwidth', 200)

    data_dir = 'training_data+sim'
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
            angle = int(filename_split[1])  # 092 part of video01_143_092.png is the angle. 90 is go straight
            speed = int(filename_split[2])
            #steering_angles.append(angle)
            car_speed.append(speed)
            #angles_speed.append([angle, speed])

    X_train, X_valid, y_train, y_valid = train_test_split(image_paths, car_speed, test_size=0.2, shuffle= True)
    print("Training data: %d\nValidation data: %d" % (len(X_train), len(X_valid)))

    y_train = np.array(y_train)/35
    y_valid = np.array(y_valid)/35

    model = nvidia_model()
    print(model.summary())

    # saves the model weights after each epoch if the validation loss decreased
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('lane_navigation_check_speedPredition.h5')

    history = model.fit_generator(image_data_generator(X_train, y_train, batch_size=100, is_training=True),
                                  steps_per_epoch=300,
                                  epochs=15,
                                  validation_data=image_data_generator(X_valid, y_valid, batch_size=len(X_valid), is_training= False),
                                  validation_steps=1,
                                  verbose=1,
                                  shuffle=1,
                                  callbacks=[checkpoint_callback])

    # always save model output as soon as model finishes training
    model.save('lane_navigation_final_speedPredition.h5')

    plt.plot(history.history['loss'], color='blue')
    plt.plot(history.history['val_loss'], color='red')
    plt.legend(["training loss", "validation loss"])
    plt.show()

    plt.plot(history.history['accuracy'], color='blue')
    plt.plot(history.history['val_accuracy'], color='red')
    plt.show()

if __name__ == '__main__':
    main()