import numpy as np
from numpy.random import *
import os
from utils import *
from equations import Lorenz as Lr
import matplotlib.pyplot as plt

# dataディレクトリからデータの読み込み
current_path = os.getcwd()
file_dir = current_path + "/npydata/"


def input_data(file_dir):
    f_list = np.load(file_dir + 'f_list.npy')
    f0_list = np.load(file_dir + 'f0_list.npy')
    df_dt_list = np.load(file_dir + 'df_dt_list.npy')
    vx_list = np.load(file_dir + 'vx_list.npy')
    vy_list = np.load(file_dir + 'vy_list.npy')
    vz_list = np.load(file_dir + 'vz_list.npy')
    df_dvx_list = np.load(file_dir + 'df_dvx_list.npy')
    df_dvy_list = np.load(file_dir + 'df_dvy_list.npy')
    df_dvz_list = np.load(file_dir + 'df_dvz_list.npy')
    d2f_dvxx_list = np.load(file_dir + 'd2f_dvxx_list.npy')
    d2f_dvxy_list = np.load(file_dir + 'd2f_dvxy_list.npy')
    d2f_dvyy_list = np.load(file_dir + 'd2f_dvyy_list.npy')
    d2f_dvyz_list = np.load(file_dir + 'd2f_dvyz_list.npy')
    d2f_dvzz_list = np.load(file_dir + 'd2f_dvzz_list.npy')
    d2f_dvzx_list = np.load(file_dir + 'd2f_dvzx_list.npy')

    f_list = f_list[::2, ::5, ::5, ::5]
    f0_list = f0_list[::2, ::5, ::5, ::5]
    df_dt_list = df_dt_list[::2, ::5, ::5, ::5]
    vx_list = vx_list[::2, ::5, ::5, ::5]
    vy_list = vy_list[::2, ::5, ::5, ::5]
    vz_list = vz_list[::2, ::5, ::5, ::5]
    df_dvx_list = df_dvx_list[::2, ::5, ::5, ::5]
    df_dvy_list = df_dvy_list[::2, ::5, ::5, ::5]
    df_dvz_list = df_dvz_list[::2, ::5, ::5, ::5]
    d2f_dvxx_list = d2f_dvxx_list[::2, ::5, ::5, ::5]
    d2f_dvxy_list = d2f_dvxy_list[::2, ::5, ::5, ::5]
    d2f_dvyy_list = d2f_dvyy_list[::2, ::5, ::5, ::5]
    d2f_dvyz_list = d2f_dvyz_list[::2, ::5, ::5, ::5]
    d2f_dvzz_list = d2f_dvzz_list[::2, ::5, ::5, ::5]  # もしかしたらzxにしていたかも？？要チェック
    d2f_dvzx_list = d2f_dvzx_list[::2, ::5, ::5, ::5]

    f_list = f_list.ravel()
    f0_list = f0_list.ravel()
    df_dt_list = df_dt_list.ravel()
    vx_list = vx_list.ravel()
    vy_list = vy_list.ravel()
    vz_list = vz_list.ravel()
    df_dvx_list = df_dvx_list.ravel()
    df_dvy_list = df_dvy_list.ravel()
    df_dvz_list = df_dvz_list.ravel()
    d2f_dvxx_list = d2f_dvxx_list.ravel()
    d2f_dvxy_list = d2f_dvxy_list.ravel()
    d2f_dvyy_list = d2f_dvyy_list.ravel()
    d2f_dvyz_list = d2f_dvyz_list.ravel()
    d2f_dvzz_list = d2f_dvzz_list.ravel()
    d2f_dvzx_list = d2f_dvzx_list.ravel()

    return f_list, f0_list, df_dt_list, vx_list, vy_list, vz_list, df_dvx_list, df_dvy_list\
        , df_dvz_list, d2f_dvxx_list, d2f_dvxy_list, d2f_dvyy_list, d2f_dvyz_list, d2f_dvzz_list, d2f_dvzx_list

def zscore_dX(dX):
    dXshape = dX.shape
    dX_norm_coef = np.zeros((1,2))
    dX_norm = np.zeros((1,dXshape[1]))
    for i in range(dXshape[0]):
        dXmean = dX[i].mean()
        dXstd = dX[i].std()

        zscore_dX = (dX[i] - dXmean)/dXstd
        dX_norm_coef = np.concatenate([dX_norm_coef, np.array([[dXmean, dXstd]])], axis=0)
        dX_norm = np.concatenate((dX_norm, np.array([zscore_dX])), axis=0)

    dX_norm_coef = np.delete(dX_norm_coef, 0, 0)
    dX_norm = np.delete(dX_norm, 0, 0)
    return dX_norm, dX_norm_coef

f_list, f0_list, df_dt_list, vx_list, vy_list, vz_list\
    , df_dvx_list, df_dvy_list, df_dvz_list, d2f_dvxx_list\
     , d2f_dvxy_list, d2f_dvyy_list, d2f_dvyz_list, d2f_dvzz_list, d2f_dvzx_list = input_data(file_dir)

Xlist = np.array([f_list, f0_list, df_dvx_list, df_dvy_list\
        , df_dvz_list, d2f_dvxx_list, d2f_dvxy_list, d2f_dvyy_list, d2f_dvyz_list, d2f_dvzz_list, d2f_dvzx_list])

#   dX/dtの項はこれしかない
dXlist = np.array([df_dt_list])
dXlist, dX_norm_coef = zscore_dX(dXlist)

# Libraryの作成、標準化
Theta, Theta_norm_coef = ct.CreateTheta_normal(Xlist, 2)
Xi = sd.SparsifyDynamics(Theta, dXlist,0.1)

#推定されたdXdt
infer_dXdt = np.dot(Xi, Theta)

#plt.plot(dXlist[0], alpha=0.3)
#plt.savefig("result/true.png")
#plt.show()
#plt.plot(infer_dXdt[0], alpha=0.3)
#plt.savefig("result/infer.png")
#plt.show()


#for i in range(0):
#    print(Xi[i][:-1])
print(Xi[0][:-1])

#   Xiの有次元バージョンを作成(まずは容量の確保)
Xi_dim = np.zeros(Xi.shape)
#   1.0の係数
Xi_dim[0][0] = dX_norm_coef[0][0]
for i in range(1, Xi.shape[1]):
    Xi_dim[0][0] -= dX_norm_coef[0][1]*Xi[0][i]*Theta_norm_coef[i][0]/Theta_norm_coef[i][1]
#   他の係数
for i in range(1, Xi.shape[1]):
    Xi_dim[0][i] = dX_norm_coef[0][1]*Xi[0][i]/Theta_norm_coef[i][1]

print(Xi_dim[0][:-1])

np.savetxt('result/Xi.txt',Xi)
np.savetxt('result/Xi_dim.txt',Xi_dim)
