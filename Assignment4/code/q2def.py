import numpy as np
import glob
import csv
import cv2
import matplotlib.pyplot as plt

detector_csv_path = '../data/test/results/*detector*.csv'
depth_csv_path = '../data/test/results/*depth*.csv'
img1_rows_lists = [6, 2, 0]
img2_rows_lists = [4, 1, 1]
img3_rows_lists = [3, 0, 0]
f = 721.537700/10
T = np.power(3, 2)
p_x = 609.559300/10
p_y = 172.854000/10
all_num = np.sum(img1_rows_lists) + np.sum(img2_rows_lists) + np.sum(img3_rows_lists)
font = cv2.FONT_HERSHEY_DUPLEX

def read_all_csv(path, initial_matrix):
    all_file_names = glob.glob(path)
    for filename in all_file_names:
        reader = csv.reader(open(filename, "rt"), delimiter=",")
        x = list(reader)
        result = np.array(x).astype("float")
        initial_matrix.append(result)
    return initial_matrix


def calculate_mass_center(detector_matrix, depth_matrix, img, type):

    dis_list = []

    x_list = np.arange(0, depth_matrix.shape[1], 1)
    y_list = np.arange(0, depth_matrix.shape[0], 1)
    x_vector = np.asarray(x_list).reshape(1, (len(x_list)))
    y_vector = np.asarray(y_list).reshape((len(y_list), 1))
    x_matrix = np.tile(x_list, (y_vector.shape[0], 1))
    y_matrix = np.tile(y_list.T, (x_vector.shape[1],1)).T
    X_matrix = np.divide(x_matrix * depth_matrix, f) - p_x
    Y_matrix = np.divide(y_matrix * depth_matrix, f) - p_y
    # container for mass center
    return_matrix = np.zeros((detector_matrix.shape[0], 3))
    for i in range(detector_matrix.shape[0]):
        x_left = np.int(detector_matrix[i][0])
        y_top = np.int(detector_matrix[i][1])
        x_right = np.int(detector_matrix[i][2])
        y_bottom = np.int(detector_matrix[i][3])
        id = np.int(detector_matrix[i][4])
        score = np.int(detector_matrix[i][5])

        pts = np.array([[x_left, y_top],
                        [x_right, y_top],
                        [x_right, y_bottom],
                        [x_left, y_bottom]], np.int32)
        pts = pts.reshape((-1, 1, 2))

        if type == "car":
            img = cv2.polylines(img, [pts], True, (0, 0, 255))
            cv2.putText(img, 'Car', (x_left, y_top - 5), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if type == "person":
            img = cv2.polylines(img, [pts], True, (255, 0, 0))
            cv2.putText(img, 'Person', (x_left, y_top - 5), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        if type == "bicycle":
            img = cv2.polylines(img, [pts], True, (0, 255, 0))
            cv2.putText(img, 'Bicycle', (x_left, y_top - 5), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        Z = depth_matrix[y_top-1:y_bottom-1, x_left-1:x_right-1]
        X = X_matrix[y_top-1:y_bottom-1, x_left-1:x_right-1]
        Y = Y_matrix[y_top-1:y_bottom-1, x_left-1:x_right-1]

        x_median = np.median(X)
        y_median = np.median(Y)
        z_median = np.median(Z)

        how_far = np.power(np.power(x_median,2)+np.power(y_median,2)+np.power(z_median,2), 1/2)

        dis_list.append(how_far)

        return_matrix[i,0] = x_median
        return_matrix[i,1] = y_median
        return_matrix[i,2] = z_median

        # print(x_matrix.shape)
        # print(y_matrix.shape)
        # print(corresponding_depth_matrix.shape)

        for m in range(X.shape[0]):
            for n in range(X.shape[1]):
                distance = np.power((X[m, n] - x_median), 2) + \
                           np.power((Y[m, n] - y_median), 2) + \
                           np.power((Z[m, n] - z_median), 2)
                if distance < T:
                    if type == "car":
                        img[m + y_top - 1, n + x_left - 1, :] = [0, 0, 255]
                    if type =="person":
                        img[m + y_top - 1, n + x_left - 1, :] = [255, 0, 0]
                    if type == "bicycle":
                        img[m + y_top - 1, n + x_left - 1, :] = [0, 255, 0]

    return img, return_matrix, dis_list


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def main():
    img1_car =None
    img1_person = None
    img2_car = None
    img2_person = None
    img2_bicycle = None
    img3_car = None
    img1 = cv2.imread('../data/test/left/004945.jpg')
    img2 = cv2.imread('../data/test/left/004964.jpg')
    img3 = cv2.imread('../data/test/left/005002.jpg')
    detector_matrix = []
    depth_matrix = []
    detector_matrix_list = read_all_csv(detector_csv_path, detector_matrix)
    depth_matrix_list = read_all_csv(depth_csv_path, depth_matrix)

    for detector_matrix in detector_matrix_list:
        if detector_matrix.shape[0] == 6:
            img1_car = detector_matrix
        if detector_matrix.shape[0] == 2:
            img1_person = detector_matrix
        if detector_matrix.shape[0] == 1 and detector_matrix[0][0]>1000:
            img2_bicycle = detector_matrix
        if detector_matrix.shape[0] == 4:
            img2_car = detector_matrix
        if detector_matrix.shape[0] == 1 and detector_matrix[0][0]<1000:
            img2_person = detector_matrix
        if detector_matrix.shape[0] == 3:
            img3_car = detector_matrix


    img1_depth = depth_matrix_list[2]
    img2_depth = depth_matrix_list[1]
    img3_depth = depth_matrix_list[0]


    print("finish reading!")


    car_count = img1_rows_lists[0]
    person_count = img1_rows_lists[1]
    bicycle_count = img1_rows_lists[2]

    img1, img1_car_center, car_dis_list = calculate_mass_center(img1_car, img1_depth, img1, "car")
    np.savetxt("../data/test/results/004945_car_mass_center.csv", img1_car_center, delimiter=",")
    print("image 1 car's mass center is: ")
    print(img1_car_center)
    print()

    img_1, img1_person_center, person_dis_list = calculate_mass_center(img1_person, img1_depth, img1, "person")
    np.savetxt("../data/test/results/004945_person_mass_center.csv", img1_person_center, delimiter=",")
    print("image 1 person's mass center is: ")
    print((img1_person_center))
    print()

    total_list = car_dis_list + person_dis_list
    closest_index = np.argsort(total_list)
    str =  ""
    if closest_index[0] < len(car_dis_list):
        mass_center = img1_car_center[closest_index[0]]
        str = "the closest object is car, position for you is {}.".format(mass_center)
    if len(car_dis_list)<= closest_index[0] < len(person_dis_list+car_dis_list):
        mass_center = img1_person_center[closest_index[0] - car_count]
        str = "the closest object is person, position for you is {}.".format(mass_center)
    # if len(person_dis_list+car_dis_list)<= closest_index < len(bicycle_dis_list+person_dis_list+car_dis_list):
    #     str = "the closest object is bicycle."


    cv2.putText(img1, "Car number is: {}, Person number is: {} and Bicycle number is: {}"
                .format(car_count, person_count, bicycle_count), (30, 20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img1, str, (30, 40), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow("img", img1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



    car_count = img2_rows_lists[0]
    person_count = img2_rows_lists[1]
    bicycle_count = img2_rows_lists[2]

    img2, img2_car_center, car_dis_list = calculate_mass_center(img2_car, img2_depth, img2, "car")
    np.savetxt("../data/test/results/004964_car_mass_center.csv", img2_car_center, delimiter=",")
    print("image 2 car's mass center is: ")
    print(img2_car_center)
    print()

    img2, img2_person_center, person_dis_list = calculate_mass_center(img2_person, img2_depth, img2, "person")
    np.savetxt("../data/test/results/004964_person_mass_center.csv", img2_person_center, delimiter=",")
    print("image 2 person's mass center is: ")
    print(img2_person_center)
    print()

    img2, img2_bicycle_center, bicycle_dis_list = calculate_mass_center(img2_bicycle, img2_depth, img2, "bicycle")
    np.savetxt("../data/test/results/004964_bicycle_mass_center.csv", img2_bicycle_center, delimiter=",")
    print("image 2 bicycle's mass center is: ")
    print(img2_bicycle_center)
    print()

    total_list = car_dis_list + person_dis_list
    closest_index = np.argsort(total_list)
    str =  ""
    if closest_index[0] < len(car_dis_list):
        mass_center = img2_car_center[closest_index[0]]
        str = "the closest object is car, position for you is {}.".format(mass_center)
    if len(car_dis_list)<= closest_index[0] < len(person_dis_list+car_dis_list):
        mass_center = img2_person_center[closest_index[0]-car_count]
        str = "the closest object is person, position for you is {}.".format(mass_center)
    if len(person_dis_list+car_dis_list)<= closest_index[0] < len(bicycle_dis_list+person_dis_list+car_dis_list):
        mass_center = img2_bicycle_center[closest_index[0]-car_count-person_count]
        str = "the closest object is bicycle, position for you is {}.".format(mass_center)

    cv2.putText(img2, "Car number is: {}, Person number is: {} and Bicycle number is: {}"
                .format(car_count, person_count, bicycle_count), (30, 20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img2, str, (30, 40), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow("img", img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



    car_count = img3_rows_lists[0]
    person_count = img3_rows_lists[1]
    bicycle_count = img3_rows_lists[2]

    img3, img3_car_center, car_dis_list = calculate_mass_center(img3_car, img3_depth, img3, "car")
    np.savetxt("../data/test/results/005002_car_mass_center.csv", img3_car_center, delimiter=",")
    print("image 3 car's mass center is: ")
    print(img3_car_center)
    print()

    total_list = car_dis_list + person_dis_list
    closest_index = np.argsort(total_list)
    str =  ""
    if closest_index[0] < len(car_dis_list):
        mass_center = img3_car_center[closest_index[0]]
        str = "the closest object is car, position for you is {}.".format(mass_center)
    # if len(car_dis_list)<= closest_index < len(person_dis_list+car_dis_list):
    #     str = "the closest object is person."
    # if len(person_dis_list+car_dis_list)<= closest_index < len(bicycle_dis_list+person_dis_list+car_dis_list):
    #     str = "the closest object is bicycle."

    cv2.putText(img3, "Car number is: {}, Person number is: {} and Bicycle number is: {}"
                .format(car_count, person_count, bicycle_count), (30, 20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img3, str, (30, 40), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow("img", img3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
