from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_history(history, function):
    values = [function(x) for x in history]
    fig = plt.figure()
    if len(history[0]) == 2:
        ax = fig.gca(projection='3d')
        x = [row[0] for row in history]
        y = [row[1] for row in history]
        ax.plot(x, y, values, 'ro-', linewidth=2, color='red', markersize=4)
        X1 = np.arange(min(x), max(x), 0.25)
        Y1 = np.arange(min(y), max(y), 0.25)
        X1, Y1 = np.meshgrid(X1, Y1)
        Z1 = np.zeros(X1.shape)
        for i in range(len(X1)):
            for j in range(len(X1[i])):
                Z1[i][j] = function([X1[i][j], Y1[i][j]])
        ax.plot_surface(X1, Y1, Z1, alpha=0.5)
        plt.show()
    elif len(history[0]) == 1:
        plt.plot(history, values, 'ro-', linewidth=1, color='red', markersize=3)
        ma = float(max(history)[0]) + 1
        mi = float(min(history)[0]) - 1
        x = [[mi]]
        while mi < ma:
            mi += 0.01
            x.append([mi])
        y = [function(i) for i in x]
        plt.plot(x, y)
        plt.show()
    else:
        raise ValueError('Too many dimentions')
