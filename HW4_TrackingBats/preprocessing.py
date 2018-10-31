# data loading
import cv2
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, abspath, exists
import pandas as pd

'''
    preprocessing - load the frames & locations
        input:
            - img_dir: directory of the image frames
            - loc_dir: directory of the localization file
        output:
            - image_frames: list of image frames
            - loc_data: list of pandas dataframes which contains locations in each frame
'''
def preprocess(img_dir, loc_dir):
    # read data

    loc_path = abspath(loc_dir)
    loc_list = [join(loc_path, file) for file in listdir(loc_path) if isfile(join(loc_path, file)) and 'txt' in file]
    loc_list.sort()

    image_path = abspath(img_dir)
    image_list = [join(image_path, file) for file in listdir(image_path) if isfile(join(image_path, file)) and 'ppm' in file] 
    image_list.sort()



    # load data
    image_frames = []
    for file in image_list:
        img = cv2.imread(file)
        image_frames.append(img.copy())
    image_frames = np.array(image_frames)

    loc_data = []
    for file in loc_list:
        dataframe = pd.read_csv(file, names=['x', 'y'])
        loc_data.append(dataframe.copy())
    
    return image_frames, loc_data
