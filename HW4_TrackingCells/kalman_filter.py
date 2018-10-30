# Data association - bipartite matching or multiple hypothesis tracking
# global nearest neighbor standard filter
# 1. Predict the measurements and their covariances to estimate the
# validation gates. KALMAN

# 4. Perform tracking by updating the state of each object and its
# covariance from the assignment result. KALMAN

from segmentation import *
from data_association import *

def kalman_filter(tracked_objects, current_frame):
    predictions = [[1, 3], [2, 1], [5, 5]]

    measurements = find_measurements(current_frame)
    measurements = [[1, 5], [2, 4], [5, 2]]

    #assignment = data_association(predictions, measurements)
    
    return

