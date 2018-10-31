import cv2 as cv
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.animation as anim
from data_association import *

def speed(dataset):
    '''
    Takes in a dataset of columns x and y
    return the dataframe with speed deltaX and detlaY added
    '''
    deltaX = [0]
    deltaY = [0]
    for i in range(1, len(dataset)):
        deltaX.append(dataset['x'][i] - dataset['x'][i-1])
        deltaY.append(dataset['y'][i] - dataset['y'][i-1])
    dataset['deltaX'] = deltaX
    dataset['deltaY'] = deltaY
    return dataset


def kalmanFilter(image_frames, measurements):
    '''
    

    '''
    # image_frames, loc_data
    img = image_frames[0]
    # print(img.shape)
    imgX, imgY = img.shape[:2]
    # border distance, for object that are missing in measurement, decide if drop the object, by
    # x_k_p + speed > imgX - borderDistance
    borderDistance = 0


    predictions = []
    firstMeasurement = measurements[0]
    firstMeasurement = speed(firstMeasurement)
    length, noStates = firstMeasurement.shape

    deltaT = 1

    # state transition model
    A = np.array([[1, 0, deltaT, 0], [0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Q : PROCESS NOISE COVARIANCE
    # Generally constructed intuitively
    # if confident about prediction set Q to 0.
    # Example: Q = eye(n) n: # states, x, y
    Q = np.eye(noStates) # * 0.1

    # external influence B and vector u UNKNOWN


    # shape transformation vector trivial
    H = np.eye(noStates)

    # R: MEASUREMENT NOISE COVARIANCE MATRIX
    # should be 0 since using ground truth
    R = np.eye(noStates) # * 0.01

    I = np.eye(noStates)

    # process error covariance
    # many times P is diagonal, how much deviation you expect in the initialization
    # if dont know how to start use identity matrix
    # first p_k set to R
    p_k = R

    # first x_k set to firstmeasurement 
    # can test initialize to random values
    x_k = firstMeasurement.values

    for i in range(1, len(measurements)):#len(measurements) #, len(locFiles
        x_k_prev = x_k
        p_k_previous = p_k
        measurement = measurements[i]
        measurement = speed(measurement)

        x_k_predict = np.dot(x_k_prev, A) # + w_k
        p_k_predict = np.dot(np.dot(A, p_k_previous), A.T) # + Q
        p_k_predict = p_k_predict * np.eye(noStates)
        
        # data association
        # res is the INDEX OF MEASUREMENTS associated to the predict value
        # Make data association only based on X and Y, left out deltaX, deltaY
        res = data_association(x_k_predict[:, :2], measurement.values[:, :2])
        res = np.array(res)

        # check if res contains None: indexes of prediction that does not have a measurement
        # sort index inreverse
        indexes = np.where(np.array(res) == None)[0][::-1]
        for index in indexes:
            # if object out of the frame
            if (((x_k_predict[index][0] + x_k_predict[index][2]) >= imgX-borderDistance) 
            or ((x_k_predict[index][0] + x_k_predict[index][2]) <= borderDistance)
            or ((x_k_predict[index][1] + x_k_predict[index][3]) >= imgY-borderDistance)
            or ((x_k_predict[index][1] + x_k_predict[index][3]) <= borderDistance)):
                # drop the prediction
                res = np.delete(res, index)
                x_k_predict = np.delete(x_k_predict, index, axis = 0)
                x_k_prev = np.delete(x_k_prev, index, axis = 0)
            # else if occlusion happens
            else:
                # assign prediction value to measurement
                measurement.values[index] = x_k_predict[index]

        # associate measurements back to prediction so their orders are the same
        y = measurement.values[res.tolist()]

        # Kalman gain
        K = p_k_predict/(R + p_k_predict)
        K = np.nan_to_num(K, 0)
       
        # update, reconcile
        x_k = x_k_predict+ np.dot((y - x_k_predict), K)
        
        # process covariance mat update
        p_k = (I - K)* p_k_predict

        predictions.append([x_k_prev[:, :2], x_k[:, :2]])
    return predictions

# kalmanFilter(locFiles)

