
import numpy as np
import matplotlib
import math
import cv2
from os import listdir
from os.path import isfile, join, abspath, exists


def visualization(tracked_objects, images):
    for img in range(len(images)):
        cv2.imshow("Hi", images[img])
        cv2.waitKey(0)
        
    cv2.destroyAllWindows()
    return
