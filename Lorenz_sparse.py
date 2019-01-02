import numpy as np

from utils import *
from equations import Lorenz as Lr

# ローレンツ方程式
x, y, z, t = Lr.Lorenz()
Xlist = np.array([x, y, z])
dXlist = np.diff(Xlist, axis=1)
Xlist = np.delete(Xlist, -1, 1)     # 最後の行を削除してデータ数を合わせる
# print(dXlist)
# Libraryの作成
Theta = ct.CreateTheta(Xlist, 5)
Xi = sd.SparsifyDynamics(Theta, dXlist, 0.0001)
print(Xi)
