# hw2编程题 2.6
# Author:PZY
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import line_search


x0 = np.array([1.2, 1, 1, 1])
x_min = np.array([1, 1, 1, 1])
H0 = np.eye(4)

y_wolfe = []
y_sr1 = []
y_dfp = []
y_bfgs = []


def f(x):
    r = np.zeros(4)
    for i in range(1,3):
        r[2*i-2] = 10 * (x[2*i-1] - x[2*i-2]**2)
        r[2*i-1] = 1 - x[2*i-2]
    return np.sum(r**2)


def grad_f(x):
    r = np.zeros(4)
    grad = np.zeros(4)
    for i in range(1,3):
        r[2*i-2] = 10 * (x[2*i-1] - x[2*i-2]**2)
        r[2*i-1] = 1 - x[2*i-2]
        grad[2*i-2] = 2 * r[2*i-2] * (-20 * x[2*i-2]) - 2 * r[2*i-1]
        grad[2*i-1] = 2 * r[2*i-2] * 10
    return grad


# Wolfe
x = x0
y_wolfe.append(f(x0))

while f(x) >= 1e-8:
    pk = -grad_f(x)
    alpha, fc, gc, _, _, _ = line_search(f, grad_f, x, pk, c1=0.1, c2=0.9)
    if alpha is None:
        print("Wolfe Line Search did not converge!")
        break

    x = x + alpha * pk
    y_wolfe.append(f(x))

plt.plot(y_wolfe)
plt.ylabel("Wolfe")
plt.show()
print("Wolfe非精确线搜终止需要"+str(len(y_wolfe))+"次迭代。")
print("最后一次结果："+str(f(x)))

# SR1
x = x0
H = H0
y_sr1.append(f(x0))

while(f(x) > 1e-8):
    pk = -np.matmul(H, grad_f(x))
    alpha, fc, gc, _, _, _ = line_search(f, grad_f, x, pk, c1=0.1, c2=0.9)
    if alpha is None:
        print("Wolfe Line Search did not converge!")
        break

    x = x + alpha * pk
    s = alpha * pk
    y = grad_f(x) - grad_f(x - alpha * pk)
    shy = s-np.dot(H, y)
    H = H + np.matmul(shy, shy.T) / np.dot(shy, y)
    # H0 = H0 + np.matmul(shy, shy.T) / np.dot(shy, y)
    y_sr1.append(f(x))

plt.plot(y_sr1)
plt.ylabel("SR1")
plt.show()
print("SR1优化算法终止需要"+str(len(y_sr1))+"次迭代。")

# DFP
x = x0
H = H0
y_dfp.append(f(x0))

while f(x) > 1e-8:
    pk = -np.matmul(H, grad_f(x))
    alpha, fc, gc, _, _, _ = line_search(f, grad_f, x, pk, c1=0.1, c2=0.9)
    if alpha is None:
        print("Wolfe Line Search did not converge!")
        break

    x = x + alpha * pk
    s = alpha * pk
    y = grad_f(x) - grad_f(x - alpha * pk)
    H = H + np.matmul(s, s.T)/np.dot(s, y) - np.matmul(np.matmul(H, y), np.matmul(y.T, H))/np.dot(y, np.matmul(H, y))
    y_dfp.append(f(x))

plt.plot(y_dfp)
plt.ylabel("DFP")
plt.show()
print("DFP优化算法终止需要"+str(len(y_dfp))+"次迭代。")

# BFGS
x = x0
H = H0
y_bfgs.append(f(x0))

while f(x) > 1e-8:
    pk = -np.matmul(H, grad_f(x))
    alpha, fc, gc, _, _, _ = line_search(f, grad_f, x, pk, c1=0.1, c2=0.9)
    if alpha is None:
        print("Wolfe Line Search did not converge!")
        break

    x = x + alpha * pk
    s = alpha * pk
    y = grad_f(x) - grad_f(x - alpha * pk)
    H_p1 = (1 + np.dot(y, np.matmul(H, y))/np.dot(s, y))*(np.matmul(s, s.T)/np.dot(s, y))
    H_p2 = (np.matmul(s, np.matmul(y.T, H)) + np.matmul(np.matmul(H, y), s.T))/np.dot(s, y)
    H = H + H_p1 - H_p2
    y_bfgs.append(f(x))

plt.plot(y_bfgs)
plt.ylabel("BFGS")
plt.show()
print("BFGS优化算法终止需要"+str(len(y_bfgs))+"次迭代。")