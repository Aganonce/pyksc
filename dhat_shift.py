import numpy as np
import warnings

warnings.filterwarnings('error')

def scale_d(x, y):
    alpha = 0

    try:
        alpha = np.dot(x, y.T) / np.dot(y, y.T)
        dist = np.linalg.norm(x - alpha * y) / np.linalg.norm(x)
    except RuntimeWarning:
        
        try:
            alpha = np.dot(x, y.T) / np.dot(y, y.T)
        except:
            alpha = 0

        dist = np.linalg.norm(x - alpha * y) / np.linalg.norm(x)



    return dist

def dhat(x, y):
    min_d = scale_d(x, y)
    L = len(y)

    dist = 0
    optshift = 0
    opty = np.zeros(L, dtype=np.int)

    for shift in range(-L, L + 1):
        if shift < 0:
            yshift = np.concatenate((y[(-shift):len(y)], np.zeros(-shift, dtype=np.int)))
        else:
            yshift = np.concatenate((np.zeros(shift, dtype=np.int), y[:(len(y) - shift)]))

        cur_d = scale_d(x, yshift)
        if cur_d <= min_d:

            optshift = shift
            opty = yshift
            min_d = cur_d

    if np.isnan(min_d):
        dist = 0
    else:
        dist = min_d

    return (dist, optshift, opty)