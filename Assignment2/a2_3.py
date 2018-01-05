import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# q3. a
# (1)
def cal_log(x, y, sigma):
    return (-1/(np.pi * np.power(sigma, 4)))\
           * (1-(np.power(x, 2)+np.power(y, 2))/(2*np.power(sigma, 2)))\
           * np.exp(-(np.power(x, 2)+np.power(y, 2))/(2*np.power(sigma, 2)))


def generate_log_kernel(sigma, tp):
    max = cal_log(0, 0, sigma)
    threshold = np.abs(max * tp)
    y = 0
    x = 0

    while np.abs(cal_log(x, y, sigma)) > threshold:
        x += 1
    print(x)
    height, width = 2*x-1, 2*x-1
    kernel = np.zeros((height, width))
    for i in range(x):
        for j in range(x):
            entry_val = cal_log(i, j, sigma)
            kernel[-i+x-1, -j+x-1] = entry_val
            kernel[i+x-1, -j+x-1] = entry_val
            kernel[-i+x-1, j+x-1] = entry_val
            kernel[i+x-1, j+x-1] = entry_val
    return kernel


def svd(M):
    u, s, v = np.linalg.svd(M)
    s = s.tolist()
    if(np.nonzero(s)) != 1:
        return False
    return True


# q3. b
# (2)
def cal_gaussian(mean, sigma, x):
    result = np.divide(1, np.sqrt(2 * np.pi * np.power(sigma, 2))) *  \
    np.exp(- np.divide(np.power((x-mean), 2), (2 * np.power(sigma, 2))))
    return result


def generate_gaussian_kernel(sigma, k, increment):
    x_list = np.arange(-k, k, increment)
    mean = np.mean(x_list)
    y_list = []
    for x in x_list:
        y_list.append(cal_gaussian(mean, sigma, x))
    return y_list


def cal_1d_log(sigma, x):
    result = (np.divide(np.power(x, 2), np.power(sigma, 4))- np.divide(1, np.power(sigma, 2))) * \
             np.exp(- np.divide(np.power(x, 2), (2 * np.power(sigma, 2)))) * \
             np.divide(1, np.sqrt(2 * np.pi * np.power(sigma, 2)))
    return result


def generate_1d_log_kernel(sigma, k, increment):
    x_list = np.arange(-k, k, increment)
    y_list = []
    for x in x_list:
        y_list.append(cal_1d_log(sigma, x))
    return y_list, x_list


def cal_dog(g1, g2):
    return np.array(g1)-np.array(g2)


# cite: http://kestrel.nmt.edu/~raymond/software/python_notes/paper004.html
def plot(dog, log, x_list, i):
    x = x_list
    a = dog
    b = log
    plt.plot(x, a, 'b-', label='DOG')
    plt.plot(x, b, 'r--', label='LOG')
    plt.legend(loc='upper left')
    plt.xlabel('X')
    plt.ylabel('Value')
    plt.title("Scale(increasing) {}".format(i))


def plot_multiples(scales, initial_sigma, k, increment):
    plt.figure(figsize=(20, 10))
    i = 0
    for scale in scales:
        print("Iteration: {}".format(i+1))
        gaussian_kernel_2 = generate_gaussian_kernel(initial_sigma, k, increment)
        print("Gaussian kernel 2's length is: {}".format(len(gaussian_kernel_2)))
        gaussian_kernel_1 = generate_gaussian_kernel(scale*initial_sigma, k, increment)
        print("Gaussian kernel 1's length is: {}".format(len(gaussian_kernel_1)))
        log_1d_kernel, x_list = generate_1d_log_kernel(scale*initial_sigma, k, increment)
        print("1d log kernel's length is: {}".format(len(log_1d_kernel)))
        dog = cal_dog(gaussian_kernel_1, gaussian_kernel_2)
        plt.subplot(np.ceil(np.sqrt(len(scales))), np.floor(np.sqrt(len(scales))), i + 1)
        plot(dog, log_1d_kernel, x_list, i + 1)
        i += 1

if __name__ == '__main__':
    start = datetime.now()
    kernel = generate_log_kernel(1.4, 0.02)
    print(kernel)
    print("Can be svd? {}".format(svd(kernel)))
    scales = np.arange(1, 3, 0.2)
    plot_multiples(scales, 1, 15, 0.1)
    plt.title("Initial sigma = 1")
    plot_multiples(scales, 2, 15, 0.1)
    plt.title("Initial sigma = 5")
    plt.tight_layout()
    plt.show()
    print(datetime.now() - start)


