import numpy as np
import matplotlib.pyplot as plt

# 初期分布の形状だけ確認することができる
def visualize(file_dir):
    f_list = np.load(file_dir + 'f_list.npy')
    f_shape = f_list.shape

    f_vx = np.zeros(f_shape[1])
    f_vy = np.zeros(f_shape[2])
    f_vz = np.zeros(f_shape[3])
    for i in range(f_shape[1]):
        for j in range(f_shape[2]):
            for k in range(f_shape[3]):
                f_vx[i] += f_list[0][i][j][k]
                f_vy[j] += f_list[0][i][j][k]
                f_vz[k] += f_list[0][i][j][k]
    plt.plot(f_vx)
    plt.plot(f_vy)
    plt.plot(f_vz)
    plt.show()



def main():
    file_dir = "./"
    visualize(file_dir)

if __name__=='__main__':
    main()