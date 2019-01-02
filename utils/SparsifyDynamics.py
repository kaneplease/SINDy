import numpy as np
from scipy.optimize import lsq_linear

def SparsifyDynamics(Theta, dxdt, llambda):
    xdim = len(dxdt)
    par_num = len(Theta[0])

    # 初期解
    for i in range(xdim):
        xi_i = lsq_linear(Theta.T, dxdt[i])
        if i == 0:
            Xi = np.array([xi_i.x])
        else:
            Xi = np.append(Xi, np.array([xi_i.x]), axis=0)

    # 閾値以下は無視して、残ったものでもう一度回帰
    for _ in range(3): #   10回回す様にする
        for i in range(xdim):
            useID = [n for n in range(len(Xi[i])) if abs(Xi[i][n]) > llambda]
            zeroID = [n for n in range(len(Xi[i])) if abs(Xi[i][n]) <= llambda]

            #   Xiの不要な部分を全てゼロに
            for j in zeroID:
                Xi[i][j] = 0
            #   Thetaの中で必要なものだけをまとめる
            Theta_tmp = np.array([Theta[n] for n in useID])
            #   もう一度回帰
            xi_i = lsq_linear(Theta_tmp.T, dxdt[i])
            #   Xiに代入
            for n, xi_n in enumerate(useID):
                Xi[i][xi_n] = xi_i.x[n]
    return Xi


