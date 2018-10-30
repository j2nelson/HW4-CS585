
import numpy as np
import matplotlib
import math
import cv2
from os import listdir
from os.path import isfile, join, abspath, exists

from preprocessing import *
from kalman_filter import *
from visualization import *

def main():

    # load the images
    images = preprocessing()

    # objects
    tracked_objects = [[1, 3], [2, 1], [5, 5]]

    # track using kalman filter
    for img in range(len(images)):
        kalman_filter(tracked_objects, images[img])

    # create a video of the tracks in each frame 
    visualization(tracked_objects, images)

main()
    
