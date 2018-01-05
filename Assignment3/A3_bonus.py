import numpy as np
from numpy import apply_along_axis
from matplotlib import pyplot as plt
import scipy.io as sio
from camera_params import *

# get K
K = np.zeros((3, 3))
K[0, 0] = fx_d
K[1, 1] = fy_d
K[0, 2] = px_d
K[1, 2] = py_d
K[2, 2] = 1

# get P
P = np.zeros((3, 4))
P[0, 0] = 1
P[1, 1] = 1
P[2, 2] = 1
R = np.append(R, np.zeros((1, 3)), 0)
R = np.append(R, np.zeros((4, 1)), 1)
R[3, 3] = 1
T = np.append(P, np.zeros((1, 4)), 0)
T[3, 3] = 1
T[0, 3] = t_x
T[1, 3] = t_y
T[2, 3] = t_z
P = np.dot(np.dot(np.dot(K, P), R), T)


def calculate(coordinate, P):
    depth = coordinate[2]
    coordinate[2] = 1
    result = np.linalg.lstsq(P, coordinate)
    return result[0] * depth / result[0][2]


def obj_loc(P):
    load = sio.loadmat("rgbd.mat")
    depth = np.array(load.get("depth"))
    labels = load.get("labels")
    index_0 = np.where(labels == 0)
    index_1 = np.where(labels == 1)
    index_2 = np.where(labels == 2)
    index_3 = np.where(labels == 3)

    [M, N] = depth.shape
    image_3d = np.zeros((M, N, 3))
    x_2d = np.tile(np.asarray(list(range(int(-N / 2), int(N / 2)))), (M, 1))
    y_2d = np.tile(np.asarray(list(range(int(-M / 2), int(M / 2)))), (N, 1)).reshape((M, N))

    image_3d[:, :, 0] = x_2d
    image_3d[:, :, 1] = y_2d
    image_3d[:, :, 2] = depth

    image_3d = apply_along_axis(calculate, 2, image_3d, P)

    X = image_3d[:, :, 0]
    Y = image_3d[:, :, 1]
    Z = image_3d[:, :, 2]

    average_x0 = np.mean(np.take(X, index_0))
    average_y0 = np.mean(np.take(Y, index_0))
    average_z0 = np.mean(np.take(Z, index_0))

    average_x1 = np.mean(np.take(X, index_1))
    average_y1 = np.mean(np.take(Y, index_1))
    average_z1 = np.mean(np.take(Z, index_1))

    average_x2 = np.mean(np.take(X, index_2))
    average_y2 = np.mean(np.take(Y, index_2))
    average_z2 = np.mean(np.take(Z, index_2))

    average_x3 = np.mean(np.take(X, index_3))
    average_y3 = np.mean(np.take(Y, index_3))
    average_z3 = np.mean(np.take(Z, index_3))

    lst = []
    lst.append(average_x0**2 + average_y0 ** 2 + average_z0 ** 2)
    lst.append(average_x1**2 + average_y1 ** 2 + average_z1 ** 2)
    lst.append(average_x2**2 + average_y2 ** 2 + average_z2 ** 2)
    lst.append(average_x3**2 + average_y3 ** 2 + average_z3 ** 2)
    print(np.argmax(lst))
    print(np.argmax([average_y0, average_y1, average_y2, average_y3]))






if __name__ == '__main__':
    main()
