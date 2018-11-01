
import numpy as np
import matplotlib
import math
import cv2
from os import listdir
from os.path import isfile, join, abspath, exists

from preprocessing import *
from kalman_filter import *
from visualization import *
from segmentation import *

def main():

    # load the images
    image_frames = preprocessing()

    measurements = []
    for frame in image_frames:
        measurements.append(find_measurements(frame))

    tracks = kalmanFilter(image_frames, measurements)
    # print(tracks)

    # create a video of the tracks in each frame 
    visualized_frames = visualize_track(image_frames, tracks)
    output_visualization('./output/', visualized_frames)

if __name__ == '__main__':
    main()
    
