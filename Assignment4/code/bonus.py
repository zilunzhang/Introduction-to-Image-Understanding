import numpy as np
import scipy.io as spio

# Depth Intrinsic Parameters
fx_d = 5.8262448167737955e+02
fy_d = 5.8269103270988637e+02
px_d = 3.1304475870804731e+02
py_d = 2.3844389626620386e+02

# Rotation
R = [9.9997798940829263e-01, 5.0518419386157446e-03,
     4.3011152014118693e-03, -5.0359919480810989e-03,
     9.9998051861143999e-01, -3.6879781309514218e-03,
     -4.3196624923060242e-03, 3.6662365748484798e-03,
     9.9998394948385538e-01]
R = - np.asarray(R).reshape(3, 3)
R = np.linalg.inv(R.T)

# 3D Translation
t_x = 2.5031875059141302e-02
t_z = -2.9342312935846411e-04
t_y = 6.6238747008330102e-04

# set up intrinsics matrix K
K = np.identity(3)
K[0, 0] = fx_d
K[1, 1] = fy_d
K[0, 2] = px_d
K[1, 2] = py_d

# set up projection matrix P
P_j = np.identity(3)
add = np.array([[0, 0, 0]]).T
P_j = np.concatenate((P_j, add), axis=1)

# set up rotation matrix R
temp = np.array(R).reshape(3,3)
R = np.zeros((4, 4))
R[0:3, 0:3] = temp
R[3,3] = 1

# set up translation matrix T
T = np.identity(4)
T[0, 3] = t_x
T[1, 3] = t_y
T[2, 3] = t_z

# calculate P
P = np.dot(np.dot(np.dot(K, P_j), R), T)


# read info
rgbd_info = spio.loadmat("../For-Extra-Credit-Part/rgbd.mat")
z_matrix = rgbd_info.get("depth")
labels = rgbd_info.get("labels")
row_num = z_matrix.shape[0]
col_num = z_matrix.shape[1]
threshold = 0.005


def build_index_list():
    index_list = []
    for i in range(1, 8):
        index_list.append(np.argwhere(labels == i))
    return index_list


def build_three_d_matrix():
    three_d_container = np.zeros((row_num, col_num, 3))

    x_list = np.arange(0, col_num, 1)
    x_matrix = np.tile(x_list, (row_num, 1))

    y_list = np.arange(0, row_num, 1)
    y_list = y_list[::-1]
    y_matrix = np.tile(y_list, (col_num, 1)).T

    three_d_container[:, :, 0] = x_matrix
    three_d_container[:, :, 1] = y_matrix
    three_d_container[:, :, 2] = z_matrix

    result = np.apply_along_axis(calculate_X, 2, three_d_container)

    return result


def determine_object1_on_object2(three_d_matrix, object_one_indices, object_two_indices):
    object_one_num_row = object_one_indices.shape[0]
    object_two_num_row = object_two_indices.shape[0]

    new_object_one_indices = np.apply_along_axis(index_getter, 1, object_one_indices)
    new_object_two_indices = np.apply_along_axis(index_getter, 1, object_two_indices)

    container_one = np.zeros((5, object_one_num_row))
    container_two = np.zeros((5, object_two_num_row))
    container_one[:2, :] = object_one_indices.T
    container_two[:2, :] = object_two_indices.T

    for i in range(0, 3):
        container_one[i + 2, :] = np.take(three_d_matrix[:, :, i], new_object_one_indices)
        container_two[i + 2, :] = np.take(three_d_matrix[:, :, i], new_object_two_indices)

    count = 0
    # print("during count")
    for m in range(0, object_one_num_row):
        object_one = container_one[:, m]
        index_one = np.argwhere(container_two[0] == object_one[0])
        for n in range(2, 4):
            # print("m is: {} and n is: {}".format(m, n))
            index_two = np.argwhere(container_two[1] == object_one[1] - n)
            # find the intersection for two indices
            object_two_top_index = np.intersect1d(index_one, index_two)
            if len(object_two_top_index) != 0:
                object_two_top_index = container_two[:, object_two_top_index]
                difference = np.abs(object_two_top_index[4] - object_one[4])
                if difference < threshold:
                    if object_two_top_index[3] < object_one[3]:
                        count += 1
    return count


def position_info(index_list, three_d_matrix, object_one_index, object_two_index):

    # count the number, for voting.
    index_1 = index_list[object_one_index - 1]
    index_2 = index_list[object_two_index - 1]
    count_object_one_on_object_two = determine_object1_on_object2\
        (three_d_matrix, index_1, index_2)
    count_object_two_on_object_one = determine_object1_on_object2\
        (three_d_matrix, index_2, index_1)

    # do the voting
    if count_object_one_on_object_two > count_object_two_on_object_one:
        print('vote shows object{} is on object{}'.format(object_one_index, object_two_index))
    if count_object_one_on_object_two < count_object_two_on_object_one:
        print('vote shows object{} is on object{}'.format(object_two_index, object_one_index))
    if count_object_one_on_object_two == 0 and count_object_two_on_object_one == 0:
        print('There is no relationship for object{} and object{}'.format(object_one_index, object_two_index))


# some helpers
def index_getter(input_list):
    index = input_list[0] * col_num + input_list[1]
    return index


# input [x, y, Z]
def calculate_X(input_list):
    Z = input_list[2]
    input_list[2] = 1
    # P_inverse = np.linalg.inv(P)
    # XYZ_list = np.dot(P_inverse, input_list)
    # XYZ_list = np.linalg.solve(P, input_list)
    XYZ = np.linalg.lstsq(P, input_list)
    X = XYZ[0]
    coeff = Z / X[2]
    result = coeff * X
    return result


def main():
    index_list = build_index_list()
    three_d_matrix = build_three_d_matrix()
    for i in range(1, 8):
        print("i: {}".format(i))
        for j in range(1, 8):
            print("j: {}".format(j))
            if j > i:
                position_info(index_list, three_d_matrix, i, j)


if __name__ == '__main__':
    main()





