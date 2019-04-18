'''
wolfe powell 不精确一维搜索准则
1、f(x_(k)) - f(x_(k+1)) >= -c_1 * lamda * inv(g_k) * s_(k)
2、inv(g_(k+1)) * s_(k) >= c_2 * inv(g_(k)) * s_(k)
根据计算经验，常取c_1 = 0.1, c_2 = 0.5
0 < c_1 < c_2 < 1
'''

"""
函数 f(x)=100*(x(1)^2-x(2))^2+(x(1)-1)^2
梯度 g(x)=(400*(x(1)^2-x(2))*x(1)+2*(1-x(1)),-200*(x(1)^2-x(2)))
"""
from numpy import *
import sys
import matplotlib.pyplot as plt

def object_function(xk):
    f = 100 * (xk[0, 0] ** 2 - xk[1, 0]) ** 2 + (xk[0, 0] - 1) ** 2
    return f

def gradient_function(xk):
    gk = mat(
            [
                [400 * xk[0, 0] * (xk[0, 0] ** 2 - xk[1, 0]) + 2 * (xk[0, 0] - 1)],
                [-200 * (xk[0, 0] ** 2 - xk[1, 0])]
            ]
    )
    return gk

def wolfe_powell(xk, sk):

    alpha = 1.0
    a = 0.0
    b = -sys.maxsize
    c_1 = 0.1
    c_2 = 0.5
    k = 0
    while k < 100:
        k += 1
        if object_function(xk) - object_function(xk + alpha * sk) >= -c_1 * alpha * gradient_function(xk).T * sk:
            #print('满足条件1')
            if (gradient_function(xk + alpha * sk)).T * sk >= c_2 * gradient_function(xk).T * sk:
                #print('满足条件2')
                return alpha
            else:
                a = alpha
                alpha = min(2 * alpha, (alpha + b) / 2)

        else:
            b = alpha
            alpha = 0.5 * (alpha + a)
    return alpha

# BFGS变尺度算法
def BFGS(x0, eps):
    xk = x0
    gk = gradient_function(xk)
    sigma = linalg.norm(gk)
    m = shape(x0)[0]
    HK = eye(m)  # 初始HK为二阶单位阵

    sk = -1 ** HK * gk

    step = 0
    w = zeros((2, 10 ** 3))# 保存迭代把变量xk

    while sigma > eps and step < 10000:
        # w[:, step] = xk

        step += 1
        alpha = wolfe_powell(xk, sk)

        x0 = xk
        xk = xk + alpha * sk

        delta_x = xk - x0

        g0 = gk
        gk = gradient_function(xk)
        delta_g = gk - g0
        # print('delta_x为：{}, delta_g为：{}'.format(delta_x, delta_g))
        if (delta_g.T * delta_x > 0):
            # HK = HK - (HK * delta_x * delta_x.T * HK) / (delta_x.T * HK * delta_x) + (delta_g * delta_g.T) / (delta_g.T * delta_x)
            miu = ([[1, 1], [1, 1]] + delta_g.T * HK * delta_g / (delta_x.T * delta_g))
            fenzi = miu * delta_x * delta_x.T - HK * delta_g * delta_x.T - delta_x * delta_g.T * HK
            fenmu = delta_x.T * delta_g
            HK = HK + fenzi / fenmu

        sk = -1 * HK * gk
        sigma = linalg.norm(delta_x)

        print('--The {}-th iter,sigma is {}, the result is {},object value is {:.4f}'.format(step, sigma, xk.T, object_function(xk)))
    return w

if __name__ == '__main__':
    eps = 1e-5
    x0 = mat([[0.001], [0]])

    # 变尺度算法
    W = BFGS(x0, eps)

