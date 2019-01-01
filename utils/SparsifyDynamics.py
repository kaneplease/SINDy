import numpy as np
from scipy.optimize import lsq_linear

def SparsifyDynamics(Theta, dxdt, llambda):
    xdim = len(dxdt)
    Xi = np.array([])

    # 初期解
    for i in range(xdim):
        xi_i = lsq_linear(Theta, dxdt[i])
        Xi = np.append(Xi, xi_i, axis=0)



