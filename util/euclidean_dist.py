import math
import numpy as np
# calculate euclidean distance for vectors x and y
def euclidean_dist(x, y):
    return np.linalg.norm(x-y)
    # check length
    if (len(x) is not len(y)):
        raise Exception('Vector length must align!')

    dist = 0

    # calculate euclidean distance
    for xi, yi in zip(x, y):
        dist += (xi - yi) ** 2

    dist = math.sqrt(dist)

    return dist