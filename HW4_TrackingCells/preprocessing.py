import numpy as np
import cv2
from os import listdir
from os.path import isfile, join, abspath, exists

def preprocessing():
    # read image paths
    data_path = abspath('./CS585-CellImages/Normalized')
    data_list = [join(data_path, file) for file in sorted(listdir(data_path)) if isfile(join(data_path, file)) and 'jpg' in file]

    # load images
    images = []
    for file in data_list:
        img = cv2.imread(file, 0)
        images.append(img.copy())
    images = np.array(images)

    return images
