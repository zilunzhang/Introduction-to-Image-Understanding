from PIL import Image
from Homography import *
import scipy.spatial.distance as sp
import os
from sympy.utilities.iterables import multiset_permutations

color_target = cv2.imread("findBook.jpg")
RGB_color_target = cv2.cvtColor(color_target, cv2.COLOR_BGR2RGB)
row, col, height = RGB_color_target.shape
print(RGB_color_target.shape)

old_im = Image.open("findBook.jpg")
old_size = old_im.size
new_size = (2000, 1500)
new_im = Image.new("RGB", new_size)
new_im.paste(old_im, (int((new_size[0]-old_size[0])/2),
                     int((new_size[1]-old_size[1])/2)))
new_im.save('findbook_border.jpg')
border_img = cv2.imread('findbook_border.jpg')
border_img = cv2.cvtColor(border_img, cv2.COLOR_BGR2RGB)
print("border image shape's is: {}".format(border_img.shape))


def ransac(template_img, target_img):
    # define some hyperparameter
    k = 3
    P = 0.99
    p = 0.2
    ratio_threshold = 0.8
    threshold = 20
    max_num_inliner = 0
    mssd = 0
    best_transformation = None
    min_trails = int(np.divide(np.log(1-P), np.log(1-np.power(p, k))))
    match_template_kpts, match_target_kpts = extract_match_transformation(template_img, target_img, ratio_threshold)
    num_match = len(match_template_kpts)
    print("number of match is: {}".format(num_match))
    index = 0
    print("min trails is: {}".format(min_trails))
    while index in range(min_trails):
        # print("outter iteration {}".format(index))
        # print(k)
        # print(num_match)
        random_indices = np.random.choice(num_match, k)
        match_template_kpts_rand = np.take(match_template_kpts, random_indices)
        match_target_kpts_rand = np.take(match_target_kpts, random_indices)
        M, match_template_kpts_rand, match_target_kpts_rand = affine_transformation(match_template_kpts_rand, match_target_kpts_rand)
        M_new = np.zeros((2, 3))
        M_new[0, 0] = M[0, 0]
        M_new[0, 1] = M[1, 0]
        M_new[0, 2] = M[4, 0]
        M_new[1, 0] = M[2, 0]
        M_new[1, 1] = M[3, 0]
        M_new[1, 2] = M[5, 0]
        num_inliner = 0
        distance = 0
        for i in range(num_match):
            # print("inner iteration is: {}".format(i))
            template_pt_x,  template_pt_y= match_template_kpts[i].pt
            target_pt_x, target_pt_y = match_target_kpts[i].pt
            before = np.dot(M_new, np.array((template_pt_x, template_pt_y, 1)).T)
            after = np.array((target_pt_x, target_pt_y)).reshape(2, 1)
            temp_distance = sp.euclidean(before, after)
            distance += np.power(temp_distance, 2)
            if temp_distance < threshold:
                num_inliner += 1
            if num_inliner > max_num_inliner:
                max_num_inliner = num_inliner
                best_transformation = M_new
                mssd = distance/num_match
        index += 1

    return best_transformation, mssd, match_template_kpts, match_target_kpts


def reconstruct(img_template, img_set):
    best_mssd = float("inf")
    best_trans = np.zeros((3, 3))
    index_list = []
    best_index = []
    for p in multiset_permutations(np.arange(6)):
        index_list.append(p)
    np.random.shuffle(index_list)
    # index_list = index_list[0]

    index = 0
    for permutation in index_list:
        print("permutation number is:{} ".format(index + 1))
        print(permutation)
        # permutation = [1, 3, 2, 5, 0, 4]
        img_reconstruct = merge_image(img_set, permutation)
        reconstruct_downsample = cv2.resize(img_reconstruct, (int(row/4), int(col/4)))
        template_downsample = cv2.resize(img_template, (int(row/4), int(col/4)))
        best_transformation, mssd, match_template_kpts, match_target_kpts = \
            ransac(template_downsample, reconstruct_downsample)
        if mssd < best_mssd:
            best_mssd = mssd
            best_trans = best_transformation
            best_index = permutation
            if best_mssd == 0:
                return best_mssd, best_trans, best_index
        print("best current mssd is: {}".format(best_mssd))
        print("best current index is: {}".format(best_index))
        index += 1
    return best_mssd, best_trans, best_index


# cite: https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            images.append(img)
    return images


def merge_image(list_images, permutation):
    temp = []
    # print(len(permutation))
    for i in range(len(permutation)):
        temp.append(list_images[permutation[i]])
    result = np.concatenate(temp, axis=1)
    return result


if __name__ == '__main__':
    start = datetime.now()
    color_template = cv2.imread("book.jpg")
    RGB_color_template = cv2.cvtColor(color_template, cv2.COLOR_BGR2RGB)
    transformation, mssd, match_template_kpts, match_target_kpts = ransac(border_img, RGB_color_template)
    print("M is {}".format(transformation.shape))
    print("MSSD is: {}".format(mssd))

    RGB_color_target = cv2.warpAffine(border_img, transformation, (row, col))
    plt.imshow(RGB_color_target)
    plt.show()
    BGR_color_target = cv2.cvtColor(RGB_color_target, cv2.COLOR_RGB2BGR)
    cv2.imwrite("q2a.jpg", BGR_color_target)



    images = load_images_from_folder("shredded")
    color_template = cv2.imread("mugShot.jpg")
    RGB_color_template = cv2.cvtColor(color_template, cv2.COLOR_BGR2RGB)
    best_mssd, best_tran, best_idx = reconstruct(RGB_color_template, images)
    loaded_images = load_images_from_folder("shredded")
    best_img_reconstruct = merge_image(loaded_images, best_idx)
    plt.imshow(best_img_reconstruct)
    plt.show()
    best_img_reconstruct = cv2.cvtColor(best_img_reconstruct, cv2.COLOR_BGR2RGB)
    cv2.imwrite("q2b.jpg", best_img_reconstruct)
    print("best MSSD is: {}".format(best_mssd))
    print("best transformation is: {}".format(best_tran))
    print("best index is: {}".format(best_idx))
    print(datetime.now() - start)
