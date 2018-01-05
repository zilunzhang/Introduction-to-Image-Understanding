import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import cdist
from datetime import datetime
import sklearn.preprocessing as skl

sift = cv2.xfeatures2d.SIFT_create()


# q2.a
def extract_feature(image):
    key_point, descriptor = sift.detectAndCompute(image, None)
    # print(descriptor.shape)
    # normalize the descriptor
    descriptor = skl.normalize(descriptor, norm="l2", axis=1, copy=True)
    return key_point, descriptor


# q2. b
def match(template_key_points, template_descriptor, find_key_points, find_descriptor, threshold):
    # print("template_key_points's len is: {}".format(len(template_key_points)))
    # print("target key points's len is: {}".format(len(find_key_points)))
    # print("template descriptor's shape is: {}".format(template_descriptor.shape))
    # print("target descriptor's shape is: {}".format(find_descriptor.shape))
    # get the distance matrix
    match_matrix = cdist(template_descriptor, find_descriptor, 'euclidean')
    # print("match matrix's shape is: {}".format(match_matrix.shape))
    # sort distance matrix alone row direction
    sorted_match_matrix = np.sort(match_matrix, 1)
    # do the ratio of smallest and second smallest.
    ratio = np.divide(sorted_match_matrix[:, 0], sorted_match_matrix[:, 1])
    # get the indices of value which satisfied threshold
    desired_ratio_indices = np.argwhere(ratio < threshold)
    # get those desired value
    element_by_ratio = np.take(ratio, np.ndarray.flatten(desired_ratio_indices))
    # get the desired indices after sort.
    desired_sort_indices = np.argsort(element_by_ratio)
    # get the indices of smallest element in distance matrix
    min_vector = np.argmin(match_matrix, 1)
    # find desired target kpts
    find_key_points = np.take(find_key_points, min_vector)
    find_key_points = np.take(find_key_points, desired_ratio_indices)
    find_key_points = np.take(find_key_points, desired_sort_indices)

    # find desired template kpts
    template_key_points = np.take(template_key_points,  desired_ratio_indices)
    template_key_points = np.take(template_key_points,  desired_sort_indices)

    # flatten
    desired_ratio_indices = np.ndarray.flatten(desired_ratio_indices)
    find_key_points = np.ndarray.flatten(find_key_points)
    template_key_points = np.ndarray.flatten(template_key_points)

    # get the sorted and matched key points' distance
    match_indices_list = []
    # loop over all desired ratio indices
    for index in desired_ratio_indices:
        # get min distance's indices (row indices)
        min_dis_indices = np.argmin(match_matrix[index, :])
        # add corresponding element to indices list
        match_indices_list.append(match_matrix[index, min_dis_indices])
    match_indices_list = np.array(match_indices_list)
    # take value from indices
    match_dis_list = np.take(match_indices_list, desired_sort_indices)
    # print(match_dis_list)
    # print("template key points after: {}".format(len(template_key_points)))
    # print("target key points after: {}".format(len(find_key_points)))
    return template_key_points, find_key_points, match_dis_list


def affine_transformation(match_template_points, match_find_key_points):
    # print("match template_kpt: {}".format(match_template_points.shape))
    # print("match target kpt: {}".format(match_find_key_points.shape))
    # match_template_points = match_template_points[:k+1]
    # match_find_key_points = match_find_key_points[:k+1]
    P = np.zeros((2 * len(match_template_points), 6))
    P_prime = np.zeros((2 * len(match_template_points), 1))

    i = 0
    j = 0
    # build P, P prime
    while i < len(match_template_points):
        # print("iteration is: {}".format(i))
        template_pt = match_template_points[i]
        template_point_x, template_point_y = template_pt.pt
        find_key_pt = match_find_key_points[i].pt
        find_key_point_x, find_key_point_y = find_key_pt
        P[j, :] = [template_point_x, template_point_y, 0, 0, 1, 0]
        P[j+1, :] = [0, 0, template_point_x, template_point_y, 0, 1]
        P_prime[j] = [find_key_point_x]
        P_prime[j+1] = [find_key_point_y]
        i += 1
        j += 2
    # (X^T*X)W = X^T*y
    left = np.dot(P.T, P)
    # print("left: {}".format(left.shape))
    right = np.dot(P.T, P_prime)
    # print("right: {}".format(right.shape))
    # solve w
    M = np.linalg.solve(left, right)
    return M, match_template_points, match_find_key_points


# q2. c
def homography_transformation(match_template_points, match_find_key_points):
    # print("match template_kpt: {}".format(match_template_points.shape))
    # print("match find_kpt: {}".format(match_find_key_points.shape))
    # match_template_points = match_template_points[:k+1]
    # match_find_key_points = match_find_key_points[:k+1]
    P = np.zeros((2 * len(match_template_points), 9))

    i = 0
    j = 0
    # build A
    while i < len(match_template_points):
        print("iteration is: {}".format(i))
        template_pt = match_template_points[i]
        template_point_x, template_point_y = template_pt
        find_key_pt = match_find_key_points[i]
        find_key_point_x, find_key_point_y = find_key_pt
        P[j, :] = [template_point_x, template_point_y, 1, 0, 0, 0,
                   -template_point_x * find_key_point_x,
                   -find_key_point_x * template_point_y,
                   -find_key_point_x]
        P[j+1, :] = [0, 0, 0, template_point_x, template_point_y, 1,
                     -find_key_point_y * template_point_x,
                     -find_key_point_y * template_point_y,
                     -find_key_point_y]
        i += 1
        j += 2

    e_val, e_vec = np.linalg.eig(np.dot(P.T, P))
    desired_eval_index = np.argmin(e_val)
    desired_eval = np.min(e_val)
    print("smallest e-value is: {}".format(desired_eval))
    h = e_vec[desired_eval_index]
    h = h.reshape(3, 3)
    print("homography transformation is: {}".format(h))
    return h, match_template_points, match_find_key_points


# q2. d
def visualization(template_image, template_key_point, find_image, find_key_point, M):
    M_new = np.zeros((2, 3))
    M_new[0, 0] = M[0, 0]
    M_new[0, 1] = M[1, 0]
    M_new[0, 2] = M[4, 0]
    M_new[1, 0] = M[2, 0]
    M_new[1, 1] = M[3, 0]
    M_new[1, 2] = M[5, 0]
    M_new = M
    # compute four vertices and connect edge
    before_left_top = np.asarray([0, 0, 1])
    before_left_bottom = np.asarray([template_image.shape[1], 0, 1])
    before_right_top = np.asarray([0, template_image.shape[0], 1])
    before_right_bottom = np.asarray([template_image.shape[1],
                                      template_image.shape[0], 1])
    after_left_top = np.matmul(M_new, before_left_top.T)
    after_left_bottom = np.matmul(M_new, before_left_bottom.T)
    after_right_top = np.matmul(M_new, before_right_top.T)
    after_right_bottom = np.matmul(M_new, before_right_bottom.T)

# cite: http://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html
    pts = np.array([[after_left_top[0],  after_left_top[1]],
                    [after_right_top[0], after_right_top[1]],
                    [after_right_bottom[0], after_right_bottom[1]],
                    [after_left_bottom[0], after_left_bottom[1]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    img = cv2.polylines(find_image, [pts], True, (0, 0, 255))


# cite: http://docs.opencv.org/3.0-alpha/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
    matches1to2 = [cv2.DMatch(i, i, 1.0) for i in range(np.size(find_key_point))]
    display = cv2.drawMatches(img, find_key_point, template_image, template_key_point, matches1to2, None, flags=2)
    plt.imshow(display)
    plt.show()


def visualization_ransac(template_image, template_kpts, target_image, target_kpts, M, k):
    # template_kpts = template_kpts[:k+1]
    # target_kpts = target_kpts[:k+1]
    # compute four vertices and connect edge
    before_left_top = np.asarray([0, 0, 1])
    before_left_bottom = np.asarray([template_image.shape[1], 0, 1])
    before_right_top = np.asarray([0, template_image.shape[0], 1])
    before_right_bottom = np.asarray([template_image.shape[1],
                                      template_image.shape[0], 1])
    after_left_top = np.matmul(M, before_left_top.T)
    after_left_bottom = np.matmul(M, before_left_bottom.T)
    after_right_top = np.matmul(M, before_right_top.T)
    after_right_bottom = np.matmul(M, before_right_bottom.T)

# cite: http://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html
    pts = np.array([[after_left_top[0],  after_left_top[1]],
                    [after_right_top[0], after_right_top[1]],
                    [after_right_bottom[0], after_right_bottom[1]],
                    [after_left_bottom[0], after_left_bottom[1]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    img = cv2.polylines(target_image, [pts], True, (0, 0, 255))


# cite: http://docs.opencv.org/3.0-alpha/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
    matches1to2 = [cv2.DMatch(i, i, 1.0) for i in range(np.size(target_kpts))]
    display = cv2.drawMatches(img, target_kpts, template_image, template_kpts, matches1to2, None, flags=2)
    plt.imshow(display)
    plt.show()





# q2. e
def extract_match_transformation(RGB_template_image, RGB_target_image, ratio_t):

    grey_template_image = cv2.cvtColor(RGB_template_image, cv2.COLOR_RGB2GRAY)
    grey_template_kpt, grey_template_descriptor = extract_feature(grey_template_image)
    # print("grey template kpt's len is: {}".format(len(grey_template_kpt)))
    # print("grey template descriptor's len is: {}".format(len(grey_template_descriptor)))
    # print("RGB_target_image's shape is: {}".format(RGB_target_image.shape))
    red_target_image = RGB_target_image[:, :, 0]
    green_target_image = RGB_target_image[:, :, 1]
    blue_target_image = RGB_target_image[:, :, 2]

    red_target_kpt, red_target_descriptor = extract_feature(red_target_image)
    green_target_kpt, green_target_descriptor = extract_feature(green_target_image)
    blue_target_kpt, blue_target_descriptor = extract_feature(blue_target_image)
    # print()
    match_template_red_kpt, match_target_red_kpt, red_dis = match\
        (grey_template_kpt, grey_template_descriptor, red_target_kpt, red_target_descriptor, ratio_t)
    red_info_matrix = np.array((match_template_red_kpt, match_target_red_kpt, red_dis))
    red_info_matrix = red_info_matrix.T
    # print("red info matrix's shape is: {}".format(red_info_matrix.shape))
    # print("red")
    # print()
    match_template_green_kpt, match_target_green_kpt, green_dis = match\
        (grey_template_kpt, grey_template_descriptor, green_target_kpt, green_target_descriptor, ratio_t)
    green_info_matrix = np.array((match_template_green_kpt, match_target_green_kpt, green_dis))
    green_info_matrix = green_info_matrix.T
    # print("green info matrix's shape is: {}".format(green_info_matrix.shape))
    # print("green")
    # print()
    match_template_blue_kpt, match_target_blue_kpt, blue_dis = match\
        (grey_template_kpt, grey_template_descriptor, blue_target_kpt, blue_target_descriptor, ratio_t)
    blue_info_matrix = np.array((match_template_blue_kpt, match_target_blue_kpt, blue_dis))
    blue_info_matrix = blue_info_matrix.T
    # print("blue info matrix's shape is: {}".format(blue_info_matrix.shape))
    # print("blue")

    processing_matrix = red_info_matrix
    i = 0
    while i in range(green_info_matrix.shape[0]):
        # print("i is: {}".format(i))
        j = 0
        flag = False
        while j in range(processing_matrix.shape[0]) and flag == False:
            # print("j is: {}".format(j))
            if same_coor(green_info_matrix[i][0], processing_matrix[j][0]):
                if green_info_matrix[i][2] < processing_matrix[j][2]:
                    processing_matrix[j][1] = green_info_matrix[i][1]
                    flag = True
                else:
                    flag = True
                    pass
            else:
                # print("i, j {}".format(processing_matrix.shape))
                # print(np.array(green_info_matrix[i]).reshape(1, 3).shape)
                temp = np.array(green_info_matrix[i]).reshape(1, 3)
                processing_matrix = np.concatenate((processing_matrix, temp), axis=0)
                flag = True
                # print(processing_matrix.shape)
            j += 1
        i += 1

    m = 0
    while m in range(blue_info_matrix.shape[0]):
        # print("m is: {}".format(m))
        n = 0
        flag = False
        while n in range(processing_matrix.shape[0]) and flag == False:
            # print("n is: {}".format(n))
            if same_coor(blue_info_matrix[m][0], processing_matrix[n][0]):
                if blue_info_matrix[m][2] < processing_matrix[n][2]:
                    processing_matrix[n][1] = blue_info_matrix[m][1]
                    flag = True
                else:
                    flag = True
                    pass
            else:
                # print("m, n {}".format(processing_matrix.shape))
                # print(np.array(blue_info_matrix[m]).reshape(1, 3).shape)
                temp = np.array(blue_info_matrix[m]).reshape(1, 3)
                processing_matrix = np.concatenate((processing_matrix, temp), axis=0)
                flag = True
            n += 1
        m += 1

    # print(processing_matrix.shape)
    result_template_kpts = processing_matrix[:, 0]
    result_target_kpts = processing_matrix[:, 1]

    return np.ndarray.flatten(result_template_kpts), np.ndarray.flatten(result_target_kpts)


def same_coor(pt1, pt2):
    if pt1.pt[0] == pt2.pt[0] and pt1.pt[0] == pt2.pt[0]:
        return True
    return False


if __name__ == '__main__':
    start = datetime.now()
    print(datetime.now() - start)
