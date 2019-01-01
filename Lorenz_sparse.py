import numpy as np

from utils import *
from equations import Lorenz as Lr

# ローレンツ方程式
x, y, z, t = Lr.Lorenz()
Xlist = np.array([x, y, z])
# Libraryの作成
Theta = ct.CreateTheta(Xlist, 1)
