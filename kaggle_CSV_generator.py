import Nvidia_Model
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
    model = load_model('lane_navigation_check_CombinedDataSet(max_pool_FF).h5', custom_objects={'mse_steering': mse_steering, 'mse_speed': mse_speed})
    image_id = []

    for image_path in file_list:
        image = cv2.imread(os.path.join(data_dir, image_path))
        processed_image = Nvidia_Model.img_preprocess(image)
        X = np.asarray([processed_image])  # adds batch dimensions
        predictions.append(model.predict(X))

        filename_split = re.split('[.]', image_path)
        image_id.append(int(filename_split[0]))

    image_id = pd.DataFrame(image_id, columns=['image_id'])
    predictions = pd.DataFrame(np.concatenate(predictions), columns=['angle', 'speed'])
    output = pd.concat([image_id, predictions], axis=1, sort=False)

    output.angle = (output.angle - 50)/80
    output.speed = (output.speed - 0)/35

    output.loc[output['speed'] < 0, 'speed'] = 0
    output.loc[output['speed'] > 1, 'speed'] = 1

    output.sort_values(by=['image_id'], inplace=True)
    output.to_csv('combinedDataset-maxpool-FF.csv', index=False)

if __name__ == '__main__':
    main()