import numpy as np
from scipy.optimize import lsq_linear, leastsq
import random
import matplotlib.pyplot as plt

def SparsifyDynamics(Theta, dxdt, llambda):
    # Theta = [ x(0) x(1) ... x(n)
    #           y(0) y(1) ... y(n)
    #           z(0) z(1) ... z(n) ]
    xdim = len(dxdt)
    par_num = len(Theta[0])

    opt_num = 1  # [0: lsq_linear, 1: leastsq]

    # 初期解
    for i in range(xdim):
        # 最適化手法の選択
        if opt_num == 0:
            xi_i = lsq_linear(Theta.T, dxdt[i])
            xi_i = xi_i.x
        elif opt_num == 1:
            xi_i = leastsq_for_matrix(Theta, dxdt[i])
            xi_i = xi_i[0]
            print(xi_i)

        if i == 0:
            Xi = np.array([xi_i])
        else:
            Xi = np.append(Xi, np.array([xi_i]), axis=0)

    # 閾値以下は無視して、残ったものでもう一度回帰
    for _ in range(10): #   10回回す様にする
        for i in range(xdim):
            useID = [n for n in range(len(Xi[i])) if abs(Xi[i][n]) > llambda]
            zeroID = [n for n in range(len(Xi[i])) if abs(Xi[i][n]) <= llambda]

            #   Xiの不要な部分を全てゼロに
            for j in zeroID:
                Xi[i][j] = 0
            #   Thetaの中で必要なものだけをまとめる
            Theta_tmp = np.array([Theta[n] for n in useID])
            print(Theta_tmp.shape, dxdt[i].shape)
            #   もう一度回帰
            if opt_num == 0:
                xi_i = lsq_linear(Theta_tmp.T, dxdt[i])
                xi_i = xi_i.x
            if opt_num == 1:
                xi_i = leastsq_for_matrix(Theta_tmp, dxdt[i])
                xi_i = xi_i[0]
                print(xi_i)
            #   Xiに代入
            for n, xi_n in enumerate(useID):
                Xi[i][xi_n] = xi_i[n]
    return Xi

#   param と Theta は縦に同じ行数
def func_for_leastsq(param, Theta, dxdt_i):
    # 残渣resの配列を確保
    res = np.array([dxdt_i[i] for i in range(dxdt_i.shape[0])])
    for i in range(len(param)):
        res -= param[i] * Theta[i]
    return res

def leastsq_for_matrix(Theta, dxdt_i):
    param0 = np.array([random.random() for i in range(Theta.shape[0])])
    result = leastsq(func_for_leastsq, param0, args=(Theta, dxdt_i))
    return result
