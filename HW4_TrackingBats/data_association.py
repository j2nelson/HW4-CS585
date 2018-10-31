
import numpy as np
import math
import sys
from hungarianalgorithm import *

# Description: determine the cost matrix given two sets of points
# Inputs - set A and set B
# Output - cost matrix
def calculate_cost(set_A, set_B):
    cost_matrix = [None] * len(set_A)
    for i in range(len(set_A)):
        cost_matrix[i] = [None] * len(set_B)
        for j in range(len(set_B)):
            cost_matrix[i][j] = math.sqrt((set_A[i][0] - set_B[j][0]) ** 2 + (set_A[i][1] - set_B[j][1]) ** 2)
    return cost_matrix

# Description: first predicted_value (object tracked for the longest time) gets highest priority
# so assign smallest value distance to measurement to that prediction and so on through all predictions
# Input: cost matrix
# Output: the assignment array (predicted value index : measurement index)
def greedy(matrix):

    assignment = [None] * len(matrix)

    for i in range(len(matrix)):
        smallest_value = sys.maxsize
        for j in range(len(matrix[i])):
            if matrix[i][j] < smallest_value and j not in assignment:
                assignment[i] = j
                smallest_value = matrix[i][j]

    return assignment

# Description: Formulate the 2D assignment problem and obtain a global optimal
# solution as the best assignment hypothesis.
# Input - predicted_values array & measurement array
# Output - assignment pairs (index of predicted_value : index of measurement)
def data_association(predicted_values, measurements):

    # compute the cost for each i prediction and j measurement
    cost_matrix = calculate_cost(predicted_values, measurements)

    # matching based on cost 
    assignment = greedy(cost_matrix)

    # hungarian = Hungarian(cost_matrix)
    # hungarian.calculate()

    # assignment2 = hungarian.get_results()

    # print(assignment)
    # print(assignment2)

    return assignment
