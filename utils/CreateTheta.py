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


