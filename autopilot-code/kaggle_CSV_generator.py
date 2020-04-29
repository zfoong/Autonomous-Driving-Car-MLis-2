import Nvidia_Model
import os
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def round_to(x, base=5):
    return base * round(x/base)

def main():
    data_dir = 'test_data'
    file_list = os.listdir(data_dir)
    model = load_model('lane_navigation_final_LD_Objects.h5')
    predictions = []
    image_id = []

    for image_path in file_list:
        image = plt.imread(os.path.join(data_dir, image_path))
        processed_image = Nvidia_Model.img_preprocess(image)
        X = np.asarray([processed_image])  # adds batch dimensions
        predictions.append(model.predict(X))

        filename_split = re.split('[.]', image_path)
        image_id.append(int(filename_split[0]))

    image_id = pd.DataFrame(image_id, columns=['image_id'])
    predictions = pd.DataFrame(np.concatenate(predictions), columns=['angle', 'speed'])
    output = pd.concat([image_id, predictions], axis=1, sort=False)

    output.angle = (round_to(output.angle) - 50)/80
    output.speed = round((output.speed - 0)/35)
    output.sort_values(by=['image_id'], inplace=True)
    output.to_csv('LD-Objects-Submission.csv', index=False)

if __name__ == '__main__':
    main()