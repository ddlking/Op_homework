# hw1编程题 1.3.1
# Author:PZY
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import line_search


Q = np.array([[1, 0], [0, 10]])
fixed_alpha = 0.1
min_value = 10
f_list_precise = []
f_list_fixed = []
f_list_non_precise = []
times_0 = 10
times = times_0
num_list_precise = []
num_list_fixed = []
num_list_non_precise = []


def f(x):
    return np.dot(np.dot(x.T, Q), x)/2 + 10


def df(x):
    return np.dot(Q, x)


# 精确线搜索
def compute_alpha(x):
    d = df(x)
    alpha = np.dot(d.T, d) / np.dot(np.dot(d.T, Q), d)
    return alpha


def compute_x_precise(x):
    d = df(x)
    alpha = compute_alpha(x)
    x = x - alpha * d
    return x


def compute_iters_num_precise(x_0):
    temp_num = 0
    x = x_0
    f_list_precise.append(f(x))
    while abs(f(x) - min_value) > 10 ** (-10):
        x = compute_x_precise(x)
        f_list_precise.append(f(x))
        temp_num += 1
    if times == times_0:
        plt.plot(f_list_precise)
        plt.ylabel('f_precise')
        plt.show()
    return temp_num


def compute_x_fixed(x):
    d = df(x)
    x = x - fixed_alpha * d
    return x


def compute_iters_num_fixed(x_0):
    temp_num = 0
    x = x_0
    f_list_fixed.append(f(x))
    while abs(f(x) - min_value) > 10 ** (-10):
        x = compute_x_fixed(x)
        f_list_fixed.append(f(x))
        temp_num += 1
    if times == times_0:
        plt.plot(f_list_fixed)
        plt.ylabel('f_fixed')
        plt.show()
    return temp_num


# 非精确线搜索
def compute_x_non_precise(x):
    d = df(x)
    alpha_near = line_search(f, df, x, -d)[0]
    x = x - float(alpha_near) * d
    return x


def get_iteration_num_non_precise(x_0):
    temp_num = 0
    x = x_0
    f_list_non_precise.append(f(x))
    while abs(f(x) - min_value) > 10 ** (-10):
        x = compute_x_non_precise(x)
        f_list_non_precise.append(f(x))
        temp_num += 1
    if times == times_0:
        plt.plot(f_list_non_precise)
        plt.ylabel('f_non_precise')
        plt.show()
    return temp_num


while times >= 1:
    x0 = np.random.rand(2)
    num_precise = compute_iters_num_precise(x0)
    num_fixed = compute_iters_num_fixed(x0)
    num_non_precise = get_iteration_num_non_precise(x0)
    num_list_precise.append(num_precise)
    num_list_fixed.append(num_fixed)
    num_list_non_precise.append(num_non_precise)
    times = times - 1
plt.plot(num_list_precise)
plt.ylabel('num_precise')
plt.show()

plt.plot(num_list_fixed)
plt.ylabel('num_fixed')
plt.show()

plt.plot(num_list_non_precise)
plt.ylabel('num_non_precise')
plt.show()

print('精确线搜索迭代次数：', num_list_precise)
print('固定步长迭代次数：', num_list_fixed)
print('非精确线搜索迭代次数：', num_list_non_precise)
