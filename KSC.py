import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import time
import random

from ksc_core import ksc_core

np.random.seed(1234)

def rand_color():
    colorReturn = ''
    r = lambda: random.randint(0,255)
    colorReturn = '#%02X%02X%02X' % (r(),r(),r())
    return colorReturn

def str_num(string):
    string = string.rstrip()
    str_list = string.split('\t')
    str_list.pop(0)
    arr = np.array([])
    for i in str_list:
        if (len(i) > 0):
            try:
                j = float(i)
                arr = np.append(arr, j)
            except:
                arr = np.array([])
                break
    return arr

def KSC(k, iname, delta=3, oname="", plot=False):
    """K-SC implementation. Handles data input and output. Primary computation is performed in ksc_core.

    Parameters
    ----------
    k : int
        The number of clusters.
    iname : str
        Path to data for clustering. Data must be in the format of row pairs.
        First row is identifier (name of time series), second row is string of values representing
        y-values of time series where the index is the x-value. Second row is space delimited.
        Absolute time range of time series must all be the same.
    delta : int (default: 3)
        Threshold for absolute Euclidean distance between time series in cluster iterations. 
        Once distance < delta iterations end and centroids are set.
    oname : str (default: empty)
        Path (not including filetype) to where results will be saved. Not needed if plot is False.
    plot : bool (default: False)
        Plot centroids at oname.

    Returns
    -------
    results, centroids : tuple
        Returns two dictionaries. Results contain clustered time series
        memebers where the key is the cluster number and the value is a list containing lists of float
        values representing y-values of time series where the index is the x-value. Centroids contain
        the centroid time series where key is the cluster number and the value is a list containing the
        time series shape of the centroid.
    """

    cluster_K = k

    with open(iname) as f:
        data = f.readlines()

    X_list = []
    count = 0
    for tline in data:
        a = str_num(tline)            
        if len(a) > 0:
            count += 1
            X_list.append(a)

    X = np.array(X_list)

    X_row, X_column = X.shape

    b = X / np.tile(np.amax(X, axis=1), (X_column, 1)).T

    ksc, cent = ksc_core(X, cluster_K, delta=delta)

    results = {}
    centroids = {}
    for i in range(cluster_K):
        for j in range(len(ksc)):
            if ksc[j] == i + 1:
                if i not in results:
                    results[i] = list([X[j, :]])
                else:
                    results[i].append(list(X[j, :]))

        centroids[i] = list(cent[i, :])

    if (plot):
        # Plot members of each centroid
        figure_count = 111
        for i in range(cluster_K):
            fig, ax = plt.subplots()

            max_val = 1
            for j in range(len(ksc)):
                if ksc[j] == i + 1:
                    ax.plot(X[j, :], c = rand_color(), ls=':')
                    if (max(X[j, :]) > max_val):
                        max_val = max(X[j, :])

            max_val = max_val / 2
            maximizer = 1
            if (max(cent[i, :]) > 0):
                maximizer = (max_val / max(cent[i, :]))

            ax.plot(cent[i, :] * maximizer, c = 'black', ls='-', label = 'Cluster ' + str(i + 1))

            plt.legend()

            plt.savefig(oname + "_cluster_" + str(i) + "_mem.png")
            plt.clf()

        # Plot centroids
        fig, ax = plt.subplots()
        for i in range(cluster_K):
            if (max(cent[i,:]) > 0.9 or sum(cent[i, :]) == 0):
                continue
            else:      
                ax.plot(cent[i, :], label = 'Cluster ' + str(i + 1))

        plt.legend()

        plt.savefig(oname + "_cent.png")
        plt.clf()

    return (results, centroids)