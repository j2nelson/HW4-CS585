
import numpy as np
import matplotlib
import math
import cv2
from os import listdir
from os.path import isfile, join, abspath, exists

def visualization(tracked_objects, images):
    for img in range(len(images)):
        cv2.imwrite("./output/" + str(img) + '.jpeg', images[img])

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('./output_video.MP4', fourcc, 3.0, (640,480))
    for output in images:
        output = cv2.resize(output, (640,480))
        backtorgb = cv2.cvtColor(output,cv2.COLOR_GRAY2BGR)
        out.write(backtorgb)

    out.release()

    return
