import Nvidia_Model
import AlexNet_Speed_Predictor
import os
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import cv2
import keras.backend as K

def mse_steering(y_true, y_pred):
    return K.mean(K.square(y_pred[:, 0] - y_true[:, 0]))

def mse_speed(y_true, y_pred):
    return K.mean(K.square(y_pred[:, 1] - y_true[:, 1]))

def main():
    data_dir = 'test_data'
    file_list = os.listdir(data_dir)
    predictions = []
    speed = []
    model = load_model('lane_navigation_check_Objects+simData.h5', custom_objects={'mse_steering': mse_steering, 'mse_speed': mse_speed})
    model2 = load_model('lane_navigation_check_speedPredition.h5')
    image_id = []

    for image_path in file_list:
        image = cv2.imread(os.path.join(data_dir, image_path))
        processed_image = Nvidia_Model.img_preprocess(image)
        processed_image2 = AlexNet_Speed_Predictor.img_preprocess(image)
        X = np.asarray([processed_image])  # adds batch dimensions
        X2 = np.asarray([processed_image2])
        angle = model.predict(X)
        predictions.append(angle[0][0])
        speed.append(model2.predict(X2)[0][0])


        filename_split = re.split('[.]', image_path)
        image_id.append(int(filename_split[0]))

    image_id = pd.DataFrame(image_id, columns=['image_id'])
    d = {'angle': predictions, 'speed': speed}
    predictions = pd.DataFrame(d)
    output = pd.concat([image_id, predictions], axis=1, sort=False)

    output.angle = (output.angle - 50)/80

    output.loc[output['speed'] < 0, 'speed'] = 0
    output.loc[output['speed'] > 1, 'speed'] = 1

    output.sort_values(by=['image_id'], inplace=True)
    output.to_csv('speed.csv', index=False)

if __name__ == '__main__':
    main()