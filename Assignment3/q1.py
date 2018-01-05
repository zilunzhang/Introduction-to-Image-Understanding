from Crop_Function import Crop
from PIL import Image
from Homography import *

old_im = Image.open('shoe.jpg')
old_size = old_im.size
new_size = (2500, 2000)
new_im = Image.new("RGB", new_size)
new_im.paste(old_im, (int((new_size[0]-old_size[0])/2),
                     int((new_size[1]-old_size[1])/2)))
new_im.save('shoe_with_border.jpg')
img = cv2.imread('shoe_with_border.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("image shape is: {}".format(img.shape))
rows, cols, ch = img.shape


if __name__ == '__main__':

    start = datetime.now()

    # five-dollars: 152.4 mm width * 69.85 mm height
    # five - dollars: 576 pixel width * 264 pixel height

    color_template = cv2.imread("five_dollars.jpg")
    RGB_color_template = cv2.cvtColor(color_template, cv2.COLOR_BGR2RGB)
    template_height, template_width, chan = RGB_color_template.shape
    # color_search = cv2.imread("shoe.jpg")
    color_search = cv2.imread("shoe.jpg")
    RGB_color_search = cv2.cvtColor(color_search, cv2.COLOR_BGR2RGB)
    gray_color_search = cv2.cvtColor(RGB_color_search, cv2.COLOR_RGB2GRAY)

    # cite: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials
    # /py_imgproc/py_geometric_transformation/py_geometric_transformations.html
    # crop
    cf_1 = Crop("shoe_with_border.jpg")
    point_one = cf_1.getCoord()
    print(point_one)
    point_two = cf_1.getCoord()
    print(point_two)
    point_three = cf_1.getCoord()
    print(point_three)
    point_four = cf_1.getCoord()
    print(point_four)

    # cite: http://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html
    pts = np.array([[point_one[0],  point_one[1]],
                    [point_three[0], point_three[1]],
                    [point_four[0], point_four[1]],
                    [point_two[0], point_two[1]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    img = cv2.polylines(img, [pts], True, (0, 0, 255))
    plt.imshow(img)
    plt.show()

    img_top_left = point_one
    img_top_right = point_three
    img_bottom_left = point_two
    img_bottom_right = point_four

    match_find_key_points = np.array((img_top_left, img_top_right, img_bottom_left, img_bottom_right) )
    match_template_key_points = \
        np.array(((1200, 1200), (1200, 1200+template_width-1),
                 (1200+template_height-1, 1200), (1200+template_height-1, 1200+template_width-1)))


    h, status = cv2.findHomography(match_find_key_points, match_template_key_points)

    # h, match_template_points, match_find_key_points = homography_transformation(match_find_key_points, match_template_key_points)

    print("h is: {} ".format(h))


    RGB_color_template = cv2.warpPerspective(img, h, (cols, rows))
    plt.imshow(RGB_color_template)
    plt.show()

    # length: 2.7 : 5.3
    # width: 1.2 : 2.4
    length_ratio = 5.3/2.7
    width_ratio = 2.4/1.2
    length = length_ratio * 152.4/10
    width = width_ratio * 69.85/10

    print("shoe's length is: {} cm".format(length))
    print("shoe's width is: {} cm".format(width))
    print(datetime.now() - start)

