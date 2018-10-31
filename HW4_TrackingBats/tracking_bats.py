import cv2
import numpy as np

# import your magnificent work here
from kalman_filter import *
from visualization import *
from preprocessing import * 



def main():
    # load data, parameter types refer to preprocessing.py
    image_frames, loc_data = preprocess(img_dir='./CS585-BatImages/Gray', loc_dir='./CS585-Bats-Localization/Localization')

    tracks = kalmanFilter(image_frames, loc_data)
    # print(tracks)
    
    visualized_frames = visualize_track(image_frames, tracks)
    output_visualization('./output/', visualized_frames)

if __name__ == '__main__':
    main()