import numpy as np
from numpy.random import *
import os
from utils import *
from scipy.optimize import lsq_linear
from equations import Lorenz as Lr
import matplotlib.pyplot as plt

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

    f_list = f_list[::2, ::2, ::2, ::2]
    f0_list = f0_list[::2, ::2, ::2, ::2]
    df_dt_list = df_dt_list[::2, ::2, ::2, ::2]
    vx_list = vx_list[::2, ::2, ::2, ::2]
    vy_list = vy_list[::2, ::2, ::2, ::2]
    vz_list = vz_list[::2, ::2, ::2, ::2]
    df_dvx_list = df_dvx_list[::2, ::2, ::2, ::2]
    df_dvy_list = df_dvy_list[::2, ::2, ::2, ::2]
    df_dvz_list = df_dvz_list[::2, ::2, ::2, ::2]
    d2f_dvxx_list = d2f_dvxx_list[::2, ::2, ::2, ::2]
    d2f_dvxy_list = d2f_dvxy_list[::2, ::2, ::2, ::2]
    d2f_dvyy_list = d2f_dvyy_list[::2, ::2, ::2, ::2]
    d2f_dvyz_list = d2f_dvyz_list[::2, ::2, ::2, ::2]
    d2f_dvzz_list = d2f_dvzz_list[::2, ::2, ::2, ::2]  # もしかしたらzxにしていたかも？？要チェック
    d2f_dvzx_list = d2f_dvzx_list[::2, ::2, ::2, ::2]

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

def input_symmetry(file_dir):
    f_list, f0_list, df_dt_list, vx_list, vy_list, vz_list\
        , df_dvx_list, df_dvy_list, df_dvz_list, d2f_dvxx_list\
         , d2f_dvxy_list, d2f_dvyy_list, d2f_dvyz_list, d2f_dvzz_list, d2f_dvzx_list = input_data(file_dir)
    #   各変数には，x,y,zの順で高い次数の対称式を組み込む
    vx_list = vx_list + vy_list + vz_list
    vy_list = vx_list*vy_list + vy_list*vz_list + vz_list*vx_list
    vz_list = vx_list*vy_list*vz_list

    df_dvx_list = df_dvx_list + df_dvy_list + df_dvz_list
    df_dvy_list = df_dvx_list*df_dvy_list + df_dvy_list*df_dvz_list + df_dvz_list*df_dvx_list
    df_dvz_list = df_dvx_list*df_dvy_list*df_dvz_list

    d2f_dvxx_list = d2f_dvxx_list + d2f_dvyy_list + d2f_dvzz_list
    d2f_dvyy_list = d2f_dvxx_list*d2f_dvyy_list + d2f_dvyy_list*d2f_dvzz_list + d2f_dvzz_list*d2f_dvxx_list
    d2f_dvzz_list = d2f_dvxx_list*d2f_dvyy_list*d2f_dvzz_list

    d2f_dvxy_list = d2f_dvxy_list + d2f_dvyz_list + d2f_dvzx_list
    d2f_dvyz_list = d2f_dvxy_list*d2f_dvyz_list + d2f_dvyz_list*d2f_dvzx_list + d2f_dvzx_list*d2f_dvxy_list
    d2f_dvzx_list = d2f_dvxy_list*d2f_dvyz_list*d2f_dvzx_list

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

def Boltzmann_sparse_normalize(input_method="input_data"):
    # dataディレクトリからデータの読み込み
    current_path = os.getcwd()
    file_dir = current_path + "/npydata/"

    if input_method=="input_data":
        f_list, f0_list, df_dt_list, vx_list, vy_list, vz_list\
            , df_dvx_list, df_dvy_list, df_dvz_list, d2f_dvxx_list\
             , d2f_dvxy_list, d2f_dvyy_list, d2f_dvyz_list, d2f_dvzz_list, d2f_dvzx_list = input_data(file_dir)
    elif input_method=="input_symmetry":
        f_list, f0_list, df_dt_list, vx_list, vy_list, vz_list \
            , df_dvx_list, df_dvy_list, df_dvz_list, d2f_dvxx_list \
            , d2f_dvxy_list, d2f_dvyy_list, d2f_dvyz_list, d2f_dvzz_list, d2f_dvzx_list = input_symmetry(file_dir)
    else:
        raise Exception("Input method is invalid")

    Xlist = np.array([f_list, f0_list, df_dvx_list, df_dvy_list\
            , df_dvz_list, d2f_dvxx_list, d2f_dvxy_list, d2f_dvyy_list, d2f_dvyz_list, d2f_dvzz_list, d2f_dvzx_list])

    #   dX/dtの項はこれしかない
    dXlist = np.array([df_dt_list])
    dXlist, dX_norm_coef = zscore_dX(dXlist)

    # Libraryの作成、標準化
    Theta, Theta_norm_coef = ct.CreateTheta_normal(Xlist, 2)
    Xi = sd.SparsifyDynamics(Theta, dXlist, 0.03)

    #推定されたdXdt
    infer_dXdt = np.dot(Xi, Theta)

    plt.plot(dXlist[0], alpha=0.3)
    plt.savefig("result/true.png")
    plt.show()
    plt.plot(infer_dXdt[0], alpha=0.3)
    plt.savefig("result/infer.png")
    plt.show()


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

    return Xi

#   Boltzmann_sparse_normalizeの結果から、重要度だけを判別して、再度次元込みで回帰する
#   Sparsify_Dynamicsとは違い、XiやdXlistは[0][i]の形になってる
def Apply_normalize_result():
    # 最適化手法の選択
    opt_num = 0  # [0: lsq_linear, 1: leastsq]

    # dataディレクトリからデータの読み込み
    current_path = os.getcwd()
    file_dir = current_path + "/npydata/"

    f_list, f0_list, df_dt_list, vx_list, vy_list, vz_list\
        , df_dvx_list, df_dvy_list, df_dvz_list, d2f_dvxx_list\
         , d2f_dvxy_list, d2f_dvyy_list, d2f_dvyz_list, d2f_dvzz_list, d2f_dvzx_list = input_data(file_dir)

    Xlist = np.array([f_list, f0_list, df_dvx_list, df_dvy_list\
            , df_dvz_list, d2f_dvxx_list, d2f_dvxy_list, d2f_dvyy_list, d2f_dvyz_list, d2f_dvzz_list, d2f_dvzx_list])

    #   dX/dtの項はこれしかない
    dXlist = np.array([df_dt_list])

    # Libraryの作成、標準化
    Theta = ct.CreateTheta(Xlist, 2)

    Xi = Boltzmann_sparse_normalize(input_method="input_data")
    # Xi = np.loadtxt('result/Xi.txt')  #   txtファイルからの読み込みにすると１次元の行列になってしまう
    Xi_0_1 = np.reshape(np.array([1 if x != 0 else 0 for x in Xi[0]]), Xi.shape)

    useID = [n for n in range(len(Xi_0_1[0])) if Xi_0_1[0][n] == 1]
    zeroID = [n for n in range(len(Xi_0_1[0])) if Xi_0_1[0][n] == 0]

    #   係数Xiの不要な部分を全てゼロに
    for j in zeroID:
        Xi[0][j] = 0
    #   Thetaの中で必要なものだけをまとめる
    Theta_tmp = np.array([Theta[n] for n in useID])
    #   もう一度回帰
    if opt_num == 0:
        xi_i = lsq_linear(Theta_tmp.T, dXlist[0])
        xi_i = xi_i.x
    elif opt_num == 1:
        xi_i = sd.leastsq_for_matrix(Theta_tmp, dXlist[0])
        xi_i = xi_i[0]
    #   Xiに代入
    for n, xi_n in enumerate(useID):
        Xi[0][xi_n] = xi_i[n]

    # 推定されたdXdt
    infer_dXdt = np.dot(Xi, Theta)

    # test
    Xi_test = np.zeros(Xi.shape)    #    Xi_2 = Xi のようにかくとメモリが同じになってしまう
    Xi_test[0][1] = -4.0
    Xi_test[0][2] = 4.0
    test_dXdt = np.dot(Xi_test, Theta)


    plt.plot(dXlist[0][:110000], alpha=0.3)
    plt.savefig("result/true.png")
    plt.show()
    plt.plot(infer_dXdt[0][:110000], alpha=0.3)
    plt.savefig("result/infer.png")
    plt.show()
    plt.plot(test_dXdt[0][:110000], alpha=0.3)
    plt.savefig("result/test.png")
    plt.show()

    # for i in range(0):
    #    print(Xi[i][:-1])
    print(Xi[0][:-1])

    np.savetxt('result/Xi.txt', Xi)

def Apply_normalize_result_sym():
    # 最適化手法の選択
    opt_num = 0  # [0: lsq_linear, 1: leastsq]

    # dataディレクトリからデータの読み込み
    current_path = os.getcwd()
    file_dir = current_path + "/npydata/"

    f_list, f0_list, df_dt_list, vx_list, vy_list, vz_list\
        , df_dvx_list, df_dvy_list, df_dvz_list, d2f_dvxx_list\
         , d2f_dvxy_list, d2f_dvyy_list, d2f_dvyz_list, d2f_dvzz_list, d2f_dvzx_list = input_symmetry(file_dir)

    Xlist = np.array([f_list, f0_list, df_dvx_list, df_dvy_list\
            , df_dvz_list, d2f_dvxx_list, d2f_dvxy_list, d2f_dvyy_list, d2f_dvyz_list, d2f_dvzz_list, d2f_dvzx_list])

    #   dX/dtの項はこれしかない
    dXlist = np.array([df_dt_list])

    # Libraryの作成、標準化
    Theta = ct.CreateTheta(Xlist, 2)

    Xi = Boltzmann_sparse_normalize(input_method="input_symmetry")
    # Xi = np.loadtxt('result/Xi.txt')  #   txtファイルからの読み込みにすると１次元の行列になってしまう
    Xi_0_1 = np.reshape(np.array([1 if x != 0 else 0 for x in Xi[0]]), Xi.shape)

    useID = [n for n in range(len(Xi_0_1[0])) if Xi_0_1[0][n] == 1]
    zeroID = [n for n in range(len(Xi_0_1[0])) if Xi_0_1[0][n] == 0]

    #   係数Xiの不要な部分を全てゼロに
    for j in zeroID:
        Xi[0][j] = 0
    #   Thetaの中で必要なものだけをまとめる
    Theta_tmp = np.array([Theta[n] for n in useID])
    #   もう一度回帰
    if opt_num == 0:
        xi_i = lsq_linear(Theta_tmp.T, dXlist[0])
        xi_i = xi_i.x
    elif opt_num == 1:
        xi_i = sd.leastsq_for_matrix(Theta_tmp, dXlist[0])
        xi_i = xi_i[0]
    #   Xiに代入
    for n, xi_n in enumerate(useID):
        Xi[0][xi_n] = xi_i[n]

    # 推定されたdXdt
    infer_dXdt = np.dot(Xi, Theta)

    # test
    Xi_test = np.zeros(Xi.shape)    #    Xi_2 = Xi のようにかくとメモリが同じになってしまう
    Xi_test[0][1] = -4.0
    Xi_test[0][2] = 4.0
    test_dXdt = np.dot(Xi_test, Theta)


    plt.plot(dXlist[0][:110000], alpha=0.3)
    plt.savefig("result/true.png")
    plt.show()
    plt.plot(infer_dXdt[0][:110000], alpha=0.3)
    plt.savefig("result/infer.png")
    plt.show()
    plt.plot(test_dXdt[0][:110000], alpha=0.3)
    plt.savefig("result/test.png")
    plt.show()

    # for i in range(0):
    #    print(Xi[i][:-1])
    print(Xi[0][:-1])

    np.savetxt('result/Xi.txt', Xi)

def main():
    # Apply_normalize_result()
    Apply_normalize_result_sym()
    # Boltzmann_sparse_normalize()

if __name__=='__main__':
    main()