import numpy as np

from ksc_center import ksc as ksc_center
from dhat_shift import dhat as dhat_shift

def ksc_core(A, K, delta=3):
    """K-SC core. Handles the primary iteration during clustering process.

    Parameters
    ----------
    A : numpy matrix
        A set of time series. Each row vector A(i,:) corresponds to each time series.
    K : int
        The number of clusters.
    delta : int (default: 3)
        Threshold for absolute Euclidean distance between time series in cluster iterations. 
        Once distance < delta iterations end and centroids are set.

    Returns
    -------
    mem, cent : tuple
        Membership for each time series. mem(i) = the cluster index 
        that time series i belongs to, and a set of cluster centroids. Each row vector 
        cent(i,:) corresponds to each cluster centroids.
    """

    m, n = A.shape
    mem = np.ceil(K * np.random.rand(m, 1))
    cent = np.zeros((K, n))
    D = np.zeros((m, K))

    mem = np.squeeze(np.asarray(mem))

    break_reached = False
    for iteration in range(50):
        prev_mem = mem

        for k in range(K):
            cent[k, :] = ksc_center(mem, A, k, cent[k, :])

        for i in range(m):
            x = A[i, :]
            for k in range(K):
                y = cent[k, :]

                dist, optshift, opty = dhat_shift(x, y)
                D[i, k] = dist

        val, mem = D.min(1), np.argmin(D, axis=1)

        mem = mem + 1

        print("Iteration: " + str(iteration) + " Diff: " + str(np.linalg.norm(prev_mem - mem)))
        if np.linalg.norm(prev_mem - mem) < 3:
            break_reached = True
            break

    if not break_reached:
        print("WARNING: Runtime exceeded max. Exiting core loop and printing current results.")

    return (mem, cent)