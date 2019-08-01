import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from cycler import cycler

#   Lorenz eq.
#
#   dx/dt = -px + py
#   dy/dt = -xz + rx - y
#   dz/dt = xy - bz

# 時刻tまでのx,y,zのリストを返す
def Lorenz(init_pos=[1, 1, 1], p=10.0, r=28.0, b=8.0/3.0, t=50.0):
    dt = 0.001
    nstep = int(t/dt)

    x = [0] * (nstep + 1)
    y = [0] * (nstep + 1)
    z = [0] * (nstep + 1)
    t = [0] * (nstep + 1)

    x[0] = init_pos[0]
    y[0] = init_pos[1]
    z[0] = init_pos[2]
    t[0] = 0.0

    for i in range(nstep):
        dx = dt*(-p*x[i] + p*y[i])
        dy = dt*(-x[i]*z[i] + r*x[i] - y[i])
        dz = dt*(x[i]*y[i] - b*z[i])

        x[i+1] = x[i] + dx
        y[i+1] = y[i] + dy
        z[i+1] = z[i] + dz
        t[i+1] = (i+1)*dt

    return x, y, z, t

def main():
    # init = [1, 1, 1]
    x, y, z, t = Lorenz()

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #colors = [plt.cm.spectral(i) for i in np.linspace(0, 1, len(t))]
    #ax.set_prop_cycle(cycler('color', colors))

    #for i in range(len(t)-1):
    #    ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i],z[i+1]], lw=0.6)
    ax.plot(x, y, z, lw=0.6)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz eq.")

    plt.savefig('Lorenz.png')
    plt.show()

if __name__ == '__main__':
    main()