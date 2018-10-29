import cv2
import numpy as np

# import your magnificent work here
from visualization import *
from preprocessing import * 


def main():
    # load data, parameter types refer to preprocessing.py
    image_frames, loc_data = preprocess(img_dir='./CS585-BatImages/Gray', loc_dir='./CS585-Bats-Localization/Localization')

    #tracks = KalmanFilter(loc_data)

    
    #----------------------------dummy test------------------------------------------
    test = True
    if test:
        #This dummy test provides a simple example of using the visualization function
        
        # dummy data -> one track with always the first position in txt files
        tracks = [[pos.iloc[0].tolist() for pos in loc_data]]
        # dummy timestamp -> start from 0 to n-1
        timestamps = [[0, len(image_frames) - 1]]
        visualized_frames = visualize_track(image_frames, tracks, timestamps)

        # store the visualized frames
        output_visualization('./output/', visualized_frames)
        print('all set. cheers~ Jason')

if __name__ == '__main__':
    main()