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
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam, RMSprop
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

def detect_line_segments(image):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = 0.5 * np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(image, rho, angle, min_threshold,
                                    np.array([]), minLineLength=8, maxLineGap=4)

    return line_segments

def make_points(image, line):
    height, width = image.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = 0  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        return lane_lines

    height, width = image.shape
    left_fit = []
    right_fit = []
    ver_lines = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if (x1 == x2):
                continue

            if (y1 - y2 < 10) and (y1 - y2 > -10):
                continue
            ver_lines.append(line_segment)

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]

            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))
    image_x = display_lines(image, ver_lines)
    # cv2.imshow('vertical lines', image_x)
    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 1:
        lane_lines.append(make_points(image, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 1:
        lane_lines.append(make_points(image, right_fit_average))
    return ver_lines #lane_lines


def display_lines(image, lines, line_color=(255, 255, 255), line_width=3):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    # cv2.imshow('line_image', line_image)
    line_image = cv2.addWeighted(image, 1, line_image, 0.5, 1)
    return line_image

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height / 2):, :, :]  # remove top third of the image, as it is not relevant for lane following
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3, 3), 0)  # Blurs image
    #cv2.imshow('image', image)
    image_grey = np.uint8(image)
    #cv2.imshow('image_grey', image_grey)
    image_edge = cv2.Canny(image_grey, 200, 400)  # detects line edges (lanes)
    #cv2.imshow('canny', image_edge)
    line_segments = detect_line_segments(image_edge)  # detects line segemnts (combines edge pixels into a cohesive line)
    lane_lines = average_slope_intercept(image_edge, line_segments)
    image = display_lines(image, lane_lines)
    #cv2.imshow('final', image)
    image = cv2.resize(image, (200, 66))  # input image size (200,66) Nvidia model
    # image = image / 255  # normalizing
    return image

def nvidia_model():
    model = Sequential(name='Nvidia_Model')

    # elu=Expenential Linear Unit, similar to leaky Relu
    # skipping 1st hiddel layer (nomralization layer), as we have normalized the data

    # Convolution Layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu', data_format="channels_last"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.2))  # not in original model. added for more robustness
    model.add(Conv2D(64, (3, 3), activation='elu'))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dropout(0.2))  # not in original model. added for more robustness
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))

    # output layer: turn angle (from 45-135, 90 is straight, <90 turn left, >90 turn right)
    model.add(Dense(2, activation='linear'))

    # since this is a regression problem not classification problem,
    # we use MSE (Mean Squared Error) as loss function
    optimizer = Adam(lr=1e-3)  # lr is learning rate
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    return model


def image_data_generator(image_paths, angle_speed, batch_size, is_training):
    while True:
        batch_images = []
        batch_angles_speeds = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image_path = image_paths[random_index]
            image = plt.imread(image_paths[random_index])
            steering_angle = angle_speed[0, random_index]
            if is_training:
                #training: augment image
                image, steering_angle = random_augment(image, steering_angle)

            image = img_preprocess(image)
            batch_images.append(image)
            batch_angles_speeds.append([steering_angle, angle_speed[1, random_index]])

        yield (np.asarray(batch_images), np.array(batch_angles_speeds))

def main():
    print(f'tf.__version__: {tf.__version__}')
    print(f'keras.__version__: {keras.__version__}')
    np.set_printoptions(formatter={'float_kind': lambda x: "%.4f" % x})
    pd.set_option('display.width', 300)
    pd.set_option('display.float_format', '{:,.4f}'.format)
    pd.set_option('display.max_colwidth', 200)

    data_dir = 'training_data_no_objects'
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
            steering_angles.append(angle)
            car_speed.append(speed)
            angles_speed.append([angle, speed])

    X_train, X_valid, y_train, y_valid = train_test_split(image_paths, angles_speed, test_size=0.2, shuffle= True)
    print("Training data: %d\nValidation data: %d" % (len(X_train), len(X_valid)))

    y_train = np.transpose(np.array(y_train))
    y_valid = np.transpose(np.array(y_valid))
    print(y_train.shape)

    model = nvidia_model()
    print(model.summary())

    # saves the model weights after each epoch if the validation loss decreased
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('lane_navigation_check_pre.h5')

    history = model.fit_generator(image_data_generator(X_train, y_train, batch_size=300, is_training=True),
                                  steps_per_epoch=300,
                                  epochs=10,
                                  validation_data=image_data_generator(X_valid, y_valid, batch_size=100, is_training= False),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle=1,
                                  callbacks=[checkpoint_callback])

    # always save model output as soon as model finishes training
    model.save('lane_navigation_final_pre.h5')

    plt.plot(history.history['loss'], color='blue')
    plt.plot(history.history['val_loss'], color='red')
    plt.legend(["training loss", "validation loss"])
    plt.show()


    from sklearn.metrics import mean_squared_error, r2_score

    def summarize_prediction(Y_true, Y_pred):
        mse = mean_squared_error(Y_true, Y_pred)
        r_squared = r2_score(Y_true, Y_pred)
        print(f'mse       = {mse:.2}')
        print(f'r_squared = {r_squared:.2%}')

    def predict_and_summarize(X, Y):
        model = load_model('lane_navigation_check_pre.h5')
        Y_pred = model.predict(X)
        summarize_prediction(Y, Y_pred)
        return Y_pred

    n_tests = 100
    X_test, y_test = next(image_data_generator(X_valid, y_valid, n_tests, False))
    y_pred = predict_and_summarize(X_test, y_test)

    plt.scatter(y_test, y_pred)
    plt.show()

if __name__ == '__main__':
    main()