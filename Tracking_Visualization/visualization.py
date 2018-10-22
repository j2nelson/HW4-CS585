import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from os import listdir, makedirs
from os.path import isfile, join, abspath, exists

import pandas as pd

import random
random.seed(1)

# read data
loc_path = abspath('./CS585-Bats-Localization/Localization')
loc_list = [join(loc_path, file) for file in listdir(loc_path) if isfile(join(loc_path, file)) and 'txt' in file]
image_path = abspath('./CS585-BatImages/Gray')
image_list = [join(image_path, file) for file in listdir(image_path) if isfile(join(image_path, file)) and 'ppm' in file] 

# load data
image_frames = []
for file in image_list:
    img = cv2.imread(file)
    image_frames.append(img.copy())
image_frames = np.array(image_frames)

loc_data = []
for file in loc_list:
    dataframe = pd.read_table(file, sep=',')
    loc_data.append(dataframe.copy())

#----------------------------preprocessing------------------------------------------

'''
    visualize_track(img_frames, track_info, timestamps)
        input:
            - img_frames: a sequence of image frames inside the video
            - track_info: a list of tracks each contains a sequence of coordinates ordered by timestamp
            - timestamps: corresponds to track_info, element[i] = [starting time, ending time] for track i
        output:
            - visualized_frames: a sequence of track-visualized image frames
'''
def visualize_track(img_frames, track_info, timestamps):
    visualized_frames = img_frames.copy()

    # generate different colors for tracks
    random_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))\
                    for _ in range(len(track_info))]
    
    for frame_index in range(len(visualized_frames)):
        for track_index in range(len(track_info)):
            # visualize the trace when the object currently exist in the frame
            if timestamps[track_index][0] < frame_index and timestamps[track_index][1] >= frame_index:
                for index in range(1, frame_index - timestamps[track_index][0] + 1):
                    cv2.line(visualized_frames[frame_index],tuple(track_info[track_index][index - 1]),tuple(track_info[track_index][index]),random_colors[track_index],2)
    
    return visualized_frames

# dummy data -> one track with always the first position in txt files
tracks = [[pos.iloc[0].tolist() for pos in loc_data]]
# dummy timestamp -> start from 0 to n-1
timestamps = [[0, len(image_frames) - 1]]
visualized_frames = visualize_track(image_frames, tracks, timestamps)



# write the image to the output file
if not exists('./output'):
    makedirs('./output')

count = 1
for output in visualized_frames:
    cv2.imwrite('./output/'+str(count)+'.jpeg',output)
    count += 1