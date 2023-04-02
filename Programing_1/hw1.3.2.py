# hw1编程题 1.3.2
# Author:PZY
import numpy as np
import matplotlib.pyplot as plt


n = 20
x_0 = np.random.rand(n)
x_list = []
delta_x = []
angle = []
delta_norm = []


def gen_matrix(n):
    B = np.random.rand(n, n)
    A = np.matmul(B, B.T)
    return A


def gradient_descent(A_in, x_in, max_iter=20):
    x = x_in
    x_list.append(x)
    for i in range(max_iter):
        y = np.dot(A_in, x)
        x = y/np.linalg.norm(y)
        x_list.append(x)


def compute_delta_x():
    x_list_array = np.array(x_list)
    for i in range(x_list_array.shape[0]):
        delta_x.append(np.linalg.norm(x_list_array[i] - x_list_array[i-1]))
    plt.plot(delta_x)
    plt.ylabel('delta_x')
    plt.show()
    return delta_x


def compute_max_eigen(A):
    eigen = np.linalg.eig(A)
    index = np.argmax(eigen[0])
    vector = eigen[1][:, index]
    vector_final = (np.real(vector)).T
    return vector_final


def compute_angle(u):
    x_list_array = np.array(x_list)
    u = u/np.linalg.norm(u)
    for i in range(x_list_array.shape[0]):
        x_list_array[i] = x_list_array[i]/np.linalg.norm(x_list_array[i])
        angle.append(np.arccos(np.dot(x_list_array[i], u)))
    plt.plot(angle)
    plt.ylabel('angle')
    plt.show()


def compute_delta_norm(u):
    x_list_array = np.array(x_list)
    for i in range(x_list_array.shape[0]):
        delta_norm.append(np.linalg.norm(x_list_array[i] - u))
    plt.plot(delta_norm)
    plt.ylabel('delta_norm')
    plt.show()


A = gen_matrix(n)
gradient_descent(A, x_0)
compute_delta_x()

vector_f = compute_max_eigen(A)
compute_delta_norm(vector_f)
compute_angle(vector_f)

print('x_k变化情况：', x_list[-1])
print('最大特征值对应特征向量：', vector_f)
print('多次实验结果表明，首先x_k收敛，其次x_k或收敛于最大特征值对应的特征向量，或收敛于最大特征值对应的特征向量相反向量。')