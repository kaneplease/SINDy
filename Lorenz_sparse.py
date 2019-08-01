import numpy as np
from numpy.random import *

from utils import *
from equations import Lorenz as Lr

# ローレンツ方程式
x, y, z, t = Lr.Lorenz()
dt = t[1] - t[0]
Xlist = np.array([x, y, z])
X_shape = Xlist.shape
rnd = np.array([[randn() for i in range(X_shape[1]-1)] for j in range(X_shape[0])])
dXlist = np.diff(Xlist, axis=1)/dt + rnd
Xlist = np.delete(Xlist, -1, 1)     # 最後の行を削除してデータ数を合わせる
# print(dXlist)
# Libraryの作成
Theta = ct.CreateTheta(Xlist, 5)
Xi = sd.SparsifyDynamics(Theta, dXlist, 0.01)

for i in range(3):
    print(Xi[i][:-1])

