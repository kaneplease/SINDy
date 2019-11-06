import numpy as np

# まだsinを入れていない、入れる必要はあるか？
def CreateTheta(x, order):
    # x = [[x[0] x[1] ... x[n]]
    #      [y[0] y[1] ... y[n]]
    #      [z[0] z[1] ... z[n]]]
    # numpy array

    n = x[0].size
    xdim = len(x)

    # Theta = np.array([])

    # order 0
    tmp = [[1.0 for j in range(n)]] 
    Theta = tmp

    # order 1
    for i in range(xdim):
        Theta = np.concatenate((Theta, [[x[i][j] for j in range(n)]]), axis=0)

    # order 2
    if order >= 2:
        for i in range(xdim):
            for j in range(i+1):    #　　木構造により場合の数を尽くす（ノート参照）
                Theta = np.concatenate((Theta, [[x[i][a] * x[j][a] for a in range(n)]]), axis=0)

    # order 3
    if order >= 3:
        for i in range(xdim):
            for j in range(i+1):
                for k in range(j+1):
                    Theta = np.concatenate((Theta, [[x[i][a] * x[j][a] * x[k][a] for a in range(n)]]), axis=0)

    # order 4
    if order >= 4:
        for i in range(xdim):
            for j in range(i+1):
                for k in range(j+1):
                    for l in range(k+1):
                        Theta = np.concatenate((Theta, [[x[i][a] * x[j][a] * x[k][a] * x[l][a] for a in range(n)]]),
                                               axis=0)

    # order 5
    if order >= 5:
        for i in range(xdim):
            for j in range(i+1):
                for k in range(j+1):
                    for l in range(k+1):
                        for m in range(l+1):
                            Theta = np.concatenate(
                                (Theta, [[x[i][a] * x[j][a] * x[k][a] * x[l][a] * x[m][a] for a in range(n)]]), axis=0)

    return Theta

#   1.0を含まないように重回帰分析
def CreateTheta_noConst(x, order):
    # x = [[x[0] x[1] ... x[n]]
    #      [y[0] y[1] ... y[n]]
    #      [z[0] z[1] ... z[n]]]
    # numpy array

    n = x[0].size
    xdim = len(x)

    # Theta = np.array([])

    # order 0
    tmp = [[1.0 for j in range(n)]]
    Theta = tmp

    # order 1
    for i in range(xdim):
        Theta = np.concatenate((Theta, [[x[i][j] for j in range(n)]]), axis=0)

    # order 2
    if order >= 2:
        for i in range(xdim):
            for j in range(i+1):    #　　木構造により場合の数を尽くす（ノート参照）
                Theta = np.concatenate((Theta, [[x[i][a] * x[j][a] for a in range(n)]]), axis=0)

    # order 3
    if order >= 3:
        for i in range(xdim):
            for j in range(i+1):
                for k in range(j+1):
                    Theta = np.concatenate((Theta, [[x[i][a] * x[j][a] * x[k][a] for a in range(n)]]), axis=0)

    # order 4
    if order >= 4:
        for i in range(xdim):
            for j in range(i+1):
                for k in range(j+1):
                    for l in range(k+1):
                        Theta = np.concatenate((Theta, [[x[i][a] * x[j][a] * x[k][a] * x[l][a] for a in range(n)]]),
                                               axis=0)

    # order 5
    if order >= 5:
        for i in range(xdim):
            for j in range(i+1):
                for k in range(j+1):
                    for l in range(k+1):
                        for m in range(l+1):
                            Theta = np.concatenate(
                                (Theta, [[x[i][a] * x[j][a] * x[k][a] * x[l][a] * x[m][a] for a in range(n)]]), axis=0)
    # 0行目を削除
    Theta = np.delete(Theta, 0, 0)
    return Theta

#   全ての係数をmu = 0, std = 1で正規化（時間軸に沿って）
def CreateTheta_normal(x, order):
    Theta = CreateTheta(x, order)

    Tshape = Theta.shape
    norm_coef = np.zeros((1,2))                     #   正規化した係数を格納
    Theta_norm = [[1.0 for j in range(Tshape[1])]]  #   データ数だけ格納、1.0が連続するので、標準化できない
    for i in range(1, Tshape[0]):
        Tmean = Theta[i].mean()
        Tstd = Theta[i].std()

        zscore_Theta = (Theta[i] - Tmean)/Tstd
        norm_coef = np.concatenate((norm_coef, [[Tmean, Tstd]]), axis=0)
        Theta_norm = np.concatenate((Theta_norm, [zscore_Theta]), axis=0)
    return Theta_norm, norm_coef






