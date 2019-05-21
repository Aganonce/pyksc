import numpy as np

from dhat_shift import dhat as dhat_shift

def ksc(mem, A, k, cur_center):
    a_list = []

    for i in range(len(mem)):
        if int(mem[i]) == k + 1:
            if np.sum(cur_center) == 0:
                opt_a = A[i, :]
            else:
                tmp, tmps, opt_a = dhat_shift(cur_center, A[i, :])

            a_list.append(opt_a.tolist())

    A_row, A_column = A.shape

    if len(a_list) == 0:
        ksc = np.zeros(A_column)
        return ksc

    a = np.array(a_list)

    a_row, a_column = a.shape
    
    if a_row == 0:
        ksc = np.zeros(A_column)
        return ksc

    check_tile = np.tile(np.sqrt(np.sum(np.power(a, 2), axis=1)), (a_column, 1)).T

    ### --------------- Catch errenous values in tile calculation (OPTIONAL) --------------- ###
    if 0 in check_tile:
        # If there are zeros in tile matrix, replace them with median calculation of other values in matrix
        m = np.median(check_tile[check_tile > 0])
        check_tile[check_tile == 0] = m

    if np.isnan(check_tile.all()):
        pass

    if np.isnan(check_tile.any()):
        pass
    ### --------------- --------------- --------------- --------------- --------------- ###

    b = a / check_tile
    M = np.dot(b.T, b) - a_row * np.eye(a_column)

    D, V = np.linalg.eig(M)

    V = V.real

    ksc = V[:, 0]
    

    if np.sum(ksc) < 0:
        ksc = -ksc

    return ksc