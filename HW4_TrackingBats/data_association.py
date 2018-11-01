
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
    threshold = 300
    assignment = [None] * len(matrix)

    for i in range(len(matrix)):
        smallest_value = sys.maxsize
        for j in range(len(matrix[i])):
            if matrix[i][j] < threshold and matrix[i][j] < smallest_value and j not in assignment:
                assignment[i] = j
                smallest_value = matrix[i][j]

    return assignment

# Description: Formulate the 2D assignment problem and obtain a global optimal
# solution as the best assignment hypothesis.
# Input - predicted_values array & measurement array
# Output - assignment array where the position is index of predicted_value
# and the value is the index of measurement
# ex. prediction 0 goes to measurement 2 would be [2]
def data_association(predicted_values, measurements):

    # compute the cost for each i prediction and j measurement
    cost_matrix = calculate_cost(predicted_values, measurements)

    # matching based on cost 
    assignment_greedy = greedy(cost_matrix)
    return assignment_greedy
    # hungarian = Hungarian(cost_matrix)
    # hungarian.calculate()

    # assignment_hungarian = hungarian.get_results()

    # final_assignment = [None] * len(assignment_hungarian)

    # for f in range(len(final_assignment)):
    #     for a in assignment_hungarian:
    #         if a[0] == f:
    #             final_assignment[f] = a[1]
    #             assignment_hungarian.remove(a)

    # return final_assignment
