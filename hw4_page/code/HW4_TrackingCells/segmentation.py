import numpy as np
import matplotlib
import math
import cv2
from os import listdir
from os.path import isfile, join, abspath, exists

def segmentation(img):
    
    _, contours_opencv, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for index in range(len(contours_opencv)):
        x, y, w, h = cv2.boundingRect(contours_opencv[index])
        bounding_boxes.append([x, y, w, h])
    
    return bounding_boxes

# Description: segment out the cells in each image and
# return the centroids as the position measurement
# Input - the current frame
# Output - the position measurements array for each cell found in the frame
def find_measurements(current_frame):
    
    measurements = []

    # convert image to binary image
    thresh = 60
    im_bw = cv2.threshold(current_frame, thresh, 255, cv2.THRESH_BINARY)[1]

    # determine the bounding box of each section
    bounding_boxes = segmentation(im_bw)
    bounding_boxes = np.array(bounding_boxes)

    # expand each bounding box by a little 
    for b in range(len(bounding_boxes)):
        x, y, w, h = bounding_boxes[b]
        cv2.rectangle(im_bw, (x, y), (x + ((int) (w*1.1)), y + ((int) (h*1.1))), 255, cv2.FILLED, 8, 0)

    # segment out the enlarged bounding boxes 
    bounding_boxes_larger = segmentation(im_bw)
    bounding_boxes_larger = np.array(bounding_boxes_larger)
    
    for b in range(len(bounding_boxes_larger)):
        x, y, w, h = bounding_boxes_larger[b]

        # only keep the larger sections
        if((w * h) < 2000):
            continue

        # draw a box around the section
        cv2.rectangle(current_frame, (x, y), (x + w, y + h), 0, 5, 8, 0)

        # draw and save the centroid of the bounding box
        centroid_x = (int) (x + w/2)
        centroid_y = (int) (y + h/2)
        cv2.circle(current_frame, (centroid_x, centroid_y), 5, 0, -1)
        measurements.append([centroid_x, centroid_y])

    return measurements


