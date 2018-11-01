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
    borderDistance = 150


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
    R = np.eye(noStates) * 0.1 # * 0.01

    I = np.eye(noStates)

    # process error covariance
    # many times P is diagonal, how much deviation you expect in the initialization
    # if dont know how to start use identity matrix
    # first p_k set to R
    p_k = R

    # first x_k set to firstmeasurement 
    # can test initialize to random values
    x_k = firstMeasurement.values
    # print('x_k initial type: ' + str(type(x_k)))
    for i in range(1, len(measurements)):#len(measurements) #, len(locFiles
        # print('frame: ' + str(i))

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
        # print("measurement shape: " + str(measurement.shape))
        # print("prediction before shape: " + str(x_k_predict.shape))
        # # print(res)
        # print(len(measurement))
        # print(len(res))


        # check if there are more predictions than measurements
        # check if res contains None: indexes of prediction that does not have a measurement
        # sort index inreverse
        # print(np.where(np.array(res) == None))
        indexes = np.where(np.array(res) == None)[0][::-1]
        # tmp = res
        # tmp = np.delete(tmp, indexes)
        # print("tmp len: "+ str(len(np.unique(tmp))))
        # print(indexes)
        # print('x_k_predict shape: ' + str(x_k_predict.shape))
        # print('measurement shape: ' + str(measurement.shape))
        # print('-' * 20)
        # print('more predictions than measurements.')
        # print("indexes length: " + str(len(indexes)))
        # print('index: ' + str(indexes))

        # print('old x_k_predict length')
        # print(len(x_k_predict))
        # print('old measurement length')
        # print(len(measurement))


        # print('index loop')
        for index in indexes:
            # print(index)
            # if object out of the frame # SPEED explodes later, so cannot evaluate based on speed
            
            # if object has no measurement
            if (((x_k_predict[index][0]) >= imgX-borderDistance) 
            or ((x_k_predict[index][0]) <= borderDistance)
            or ((x_k_predict[index][1]) >= imgY-borderDistance)
            or ((x_k_predict[index][1]) <= borderDistance)):

                # drop the prediction
                # print(res)
                res = np.delete(res, index)
                # print(res)
                # print('x_k_predict shape before delete: ' + str(x_k_predict.shape))
                x_k_predict = np.delete(x_k_predict, index, axis = 0)
                x_k_prev = np.delete(x_k_prev, index, axis = 0)
                # indexes = indexes - 1
                # print('x_k_predict shape after delete: ' + str(x_k_predict.shape))

                #  = np.delete(measurement, index, axis = 0)
            # else if occlusion happens
            # remove the unassigned measurement
            else:
                # print('out of frame, deleted')
                # drop the prediction
                # print(res)
                res = np.delete(res, index)
                # print(res)
                # print('x_k_predict shape before delete: ' + str(x_k_predict.shape))
                x_k_predict = np.delete(x_k_predict, index, axis = 0)
                x_k_prev = np.delete(x_k_prev, index, axis = 0)
                # indexes = indexes - 1
                # print('x_k_predict shape after delete: ' + str(x_k_predict.shape))

                # print('potential collision')
                # print(' measurement assigned to prediction val')
                # # assign prediction value to measurement
                # # np.insert(a, 1, [1, 2], axis=0)
                # print(type(measurement))
                # # print(measurement[0].shape)
                # print(x_k_predict[index].shape)
                # print(x_k_predict[index])
                

                # dfPrev = measurement.loc[:index]
                # dfAfter = measurement.loc[index:]

                # colNames = measurement.columns
                # measurement = measurement.values
                # if index > len(measurement):
                #     measurement = np.append(measurement, x_k_predict[index].reshape(1, -1), axis = 0)
                # else:
                #     measurement = np.insert(measurement, index, x_k_predict[index], axis=0)
                # measurement = pd.DataFrame(measurement, columns=colNames)
                # # print()
                # res[index] = index


        # print('new x_k_predict length')
        # print(len(x_k_predict))
        # print('new measurement length')
        # print(len(measurement))
        # print("res length: " + str(len(res)))
        # print(res)
        # print('-'*20)



        # if len(x_k_predict) > len(measurement):
        #     for i in range(len(x_k_predict)):
        #         if i not in res:
        #             print("i: " + str(i))
        #             # print(type(i))
        #             # print(i.shape)
        #             # print(type(indexes))
        #             # print(indexes.shape)

        #             indexes = np.append(indexes, i)
        # indexes = indexes[::-1]
        # print("new indexes: " + str(indexes))

        # associate measurements back to prediction so their orders are the same
        # print(type(res))
        # print(res.shape)
        # print(type(res[0]))
        # print(type(res.tolist()))
        # print(measurement.values[res.tolist()])
        # print(measurement.values[res])
        # print('test y')
        # print(res)
        y = measurement.values[res.tolist()]
        # print(type(y))
        # print('y fine')

        # Kalman gain
        K = p_k_predict/(R + p_k_predict)
        K = np.nan_to_num(K, 0)
       
        # update, reconcile
        # print("y shape: " + str(y.shape))
        # print("x_k_predict shape: " + str(x_k_predict.shape))
        # print("*" * 20)
        x_k = x_k_predict+ np.dot((y - x_k_predict), K)

        # process covariance mat update
        p_k = (I - K)* p_k_predict

        # visualization
        # plt.figure()
        # plt.scatter(x_k[:, 0], x_k[:, 1], marker = 'x', label = 'prediction')
        # plt.scatter(x_k[:, 0], x_k[:, 1], marker = 'x', label = 'prediction')
        # plt.legend(loc = 0)
        # plt.title(i)

        


        # print("*" * 20)
        # print('x_k updated type: ' + str(type(x_k)))
        # print(type(x_k[:, :2].astype(int)))
        # print("*" * 20)
        # print('x_k_prev updated type: ' + str(type(x_k_prev)))
        # print(type(x_k_prev[:, :2].astype(int)))
        # print("*" * 20)
        # print("*" * 20)

        prev_points = x_k_prev[:, :2].astype(int)
        present_points = x_k[:, :2].astype(int)

        predictions.append([prev_points, present_points])

        # print(x_k.type)
        #check if there are more measurements than prediction and add measurement to prediction
        if len(measurement) > len(x_k):
            for i in range(len(measurement)):
                if i not in res:
                    x_k = np.append(x_k, measurement.values[i].reshape(1, 4), axis = 0)
        # print("-"* 20)
        # print('x_k shape: ' + str(x_k.shape))
        # print('measurement shape: ' + str(measurement.shape))
        # print("-"* 20)

    return predictions

# kalmanFilter(locFiles)

