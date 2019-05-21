import numpy as np

def scale_d(x, y):
    alpha = np.dot(x, y.T) / np.dot(y, y.T)
    dist = np.linalg.norm(x - alpha * y) / np.linalg.norm(x)
    return dist