import numpy as np


w = np.array([1.1, -6.0])
b = 2
x = np.array([5, 10]).T

y = 1

def calculate_using_expression(x_i):
    minus_y = -(np.dot(w, x) + b)
    exp_term = np.exp(minus_y)
    logistic = np.divide(1, 1+exp_term)
    derivative = x_i * (logistic -y)

    print("result is: {}".format(derivative))

    return derivative


def main():
    derivative_w1 = calculate_using_expression(x[0])
    derivative_w2 = calculate_using_expression(x[1])
    derivative_b = calculate_using_expression(1)

if __name__ == "__main__":
    main()