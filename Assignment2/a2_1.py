import numpy as np
from scipy.ndimage import *
from skimage import morphology
from PIL import Image, ImageOps
import cv2
from datetime import datetime


# A2
# 1. (a)
def find_R(image, r, threshold_coeff, sigma):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    in_float = np.float64(gray)
    Ix, Iy = np.gradient(in_float)
    Ix_2 = Ix * Ix
    Iy_2 = Iy * Iy
    IxIy = Ix * Iy
    gIx_2 = gaussian_filter(Ix_2, sigma=sigma)
    gIy_2 = gaussian_filter(Iy_2, sigma=sigma)
    gIxIy = gaussian_filter(IxIy, sigma=sigma)
    print("gIx_2 shape: {}, gIy_2 shape: {}, gIxIY shape: {}".format(gIx_2.shape, gIy_2.shape, gIxIy.shape))
    alpha = 0.05
    R = gIx_2 * gIy_2 - gIxIy**2 - alpha * (gIx_2 + gIy_2)**2
    print("R shape is: {}".format(R.shape))
    R = nms(R, r)
    image[R > threshold_coeff * R.max()] = [224, 255, 255]
    print("......")
    print("displaying Q 1.a......")
    print("......")
    return image


# 1. (b)
def nms(R, r):
    circle = morphology.disk(r)
    R = maximum_filter(R, footprint=circle)
    print("......")
    print("Using Q 1.b......")
    print("......")
    return R


# 1. (c)
# input : RGB image
def search_scale_invariant_interesting_point(image, threshold, scale_factor, sigma, k_value, r):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # cite: https://stackoverflow.com/questions/2498875/how-to-invert-colors-of-image-with-pil-python-imaging
    img = Image.fromarray(gray)
    inverted_image = ImageOps.invert(img)
    in_float = np.float64(inverted_image)
    height = in_float.shape[0]
    width = in_float.shape[1]
    three_d_matrix = np.zeros((height, width, scale_factor))
    for i in range(scale_factor):
        print("iteration : {}".format(i+1))
        gol = np.power(sigma * np.power(k_value, i), 2) * gaussian_laplace(in_float, sigma = sigma * np.power(k_value, i))
        print("sigma is: {}".format(sigma * np.power(k_value, i)))
        print()
        three_d_matrix[:, :, i] = gol
    print("3d matrix's shape is: {}".format(three_d_matrix.shape))
    value_matrix = np.max(three_d_matrix, 2)
    print("value matrix's shape is: {}".format(value_matrix.shape))
    index_matrix = np.argmax(three_d_matrix, 2)
    print("index matrix's shape is: {}".format(index_matrix.shape))
    condition = nms(value_matrix, r) - value_matrix
    value_matrix[condition != 0] = 0
    interesting_points = np.argwhere(value_matrix > threshold * value_matrix.max())
    print("interesting point's shape is: {}".format(interesting_points.shape))
#    print(interesting_points)
    for i in range(interesting_points.shape[0]):
        x = interesting_points[i, 1]
        y = interesting_points[i, 0]
        # cite: https://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html
        cv2.circle(image, (x, y), np.power(2*(index_matrix[y][x] + 2), 2), (255, 245, 0), 2)
    print("displaying Q 1.c......")
    print("......")
    return image


# 1.(d)
def display_in_one():

    image_building_log = cv2.imread("building.jpg")
    RGB_image_building_log = cv2.cvtColor(image_building_log, cv2.COLOR_BGR2RGB)
    image_building_harries = cv2.imread("building.jpg")
    RGB_image_building_harries = cv2.cvtColor(image_building_harries, cv2.COLOR_BGR2RGB)

    image_synthetic_log = cv2.imread("synthetic.png")
    RGB_image_synthetic_log = cv2.cvtColor(image_synthetic_log, cv2.COLOR_BGR2RGB)
    image_synthetic_harries = cv2.imread("synthetic.png")
    RGB_image_synthetic_harries = cv2.cvtColor(image_synthetic_harries, cv2.COLOR_BGR2RGB)

    radius = 1
    # 1st graph
    building_log = search_scale_invariant_interesting_point \
        (RGB_image_building_log, threshold=0.7, scale_factor=7, sigma=5, k_value=1.9, r=20)
    building_harries = find_R(RGB_image_building_harries, radius, 0.05, 0.5)

    # 2nd graph
    synthetic_log = search_scale_invariant_interesting_point\
        (RGB_image_synthetic_log , threshold=0.54, scale_factor=7, sigma=5, k_value=1.9, r=15)
    synthetic_harries = find_R(RGB_image_synthetic_harries, 10, 0.55, 7)
    return building_log, building_harries, synthetic_log, synthetic_harries


if __name__ == '__main__':
    start = datetime.now()
#    1(a)
    image_building = cv2.imread("building.jpg")
    RGB_image_building = cv2.cvtColor(image_building, cv2.COLOR_BGR2RGB)
    r = 5
#    using 1(b)
    RGB_image_building_1 = RGB_image_building[:, :]
    one_a = find_R(RGB_image_building_1, r, 0.07, 0.5)
    one_a_format = Image.fromarray(one_a, 'RGB')
    one_a_format.show(title="1, a")

#    1(c)
    image_synthetic = cv2.imread("synthetic.png")
    RGB_image_synthetic = cv2.cvtColor(image_synthetic, cv2.COLOR_BGR2RGB)
    RGB_image_synthetic_3 = RGB_image_synthetic[:, :]
    one_c = search_scale_invariant_interesting_point\
        (RGB_image_synthetic_3, threshold=0.54, scale_factor=7, sigma=5, k_value=1.9, r=15)
    one_c_format = Image.fromarray(one_c, 'RGB')
    one_c_format.show(title="1, c")

#    1(d)
    building_log, building_harries, synthetic_log, synthetic_harries = display_in_one()

    one_d_harris_building = Image.fromarray(building_harries, 'RGB')
    one_d_harris_building.show(title="1, d, harris_building")

    one_d_harris_synthetic = Image.fromarray(synthetic_harries, 'RGB')
    one_d_harris_synthetic.show(title="1, d, harris_synthetic")

    one_d_log_building = Image.fromarray(building_log, 'RGB')
    one_d_log_building.show(title="1, d, log_building")

    one_d_log_synthetic = Image.fromarray(synthetic_log, 'RGB')
    one_d_log_synthetic.show(title="1, d, log_synthetic")

    print(datetime.now() - start)




