import numpy as np
import matplotlib.pyplot as mp
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as anim
from numpy.linalg import inv
from matplotlib import cm
import math


def parseInput(in_file_name, chk):
    temp = (np.loadtxt(in_file_name, dtype=float, ndmin=2))
    temp2 = (np.loadtxt(in_file_name, dtype=float, ndmin=2))
    if chk == False:
        return temp, temp2, temp2.reshape((1, len(temp))), temp.reshape((1, len(temp)))
    else:
        mean = temp.sum()/len(temp)
        var = np.std(temp.transpose()[0])
        temp -= mean
        temp /= var
        X = np.ones((len(temp), 2))
        X[:, 1:] = temp
        X2 = np.ones((len(temp), 2))
        X2[:, 1:] = temp2
        return X, X2, temp2.reshape((1, len(temp))), temp.reshape((1, len(temp)))


def calcTheta(X, Y):
    Xt = np.transpose(X)
    temp = inv(np.dot(Xt, X))
    return np.dot(temp, (np.dot(Xt, Y)))


def plotPts(x, y):
    mp.plot(x[0], y[0], 'o')
    axes = mp.gca()
    axes.set_xlim([min(x[0]) - 1, max(x[0]) + 1])


def plotLine(T, X, x):
    Y = [np.dot(T[0], xi) for xi in X]
    mp.plot(x[0], Y)


def plotCont(J, Th):
    fig = mp.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    x, y = zip(*Th)
    X, Y = np.meshgrid(x, y)
    Z = J*len(J)
    Z = np.array(Z)
    Z.resize((len(J), len(J)))

    ZZ = []

    def update(i):
        ZZ.append(J[i])
        XX = X[:-i+1]
        YY = Y[:-i+1]
        ax.clear()
        ax.plot_wireframe(XX, YY, ZZ)

    a = anim.FuncAnimation(fig, update, frames=len(J), repeat=False)
    mp.show()


def gradientDescent(X, Y, a, epsilon):
    T = np.zeros((1, len(X[0])))
    m = len(X)
    J = []
    T += (a/m)*(Y*X).sum(axis=0)
    Th = []
    Jold = 0
    Jnew = (Y.transpose()[0].dot(Y.transpose()[0]))/(2*m)
    while (abs(Jnew - Jold) > epsilon):
        Tx = X.dot(T.transpose())
        diff = (Y - Tx)
        pred = diff*X
        T += (a/m)*pred.sum(axis=0)
        Th.append((T[0][0], T[0][1]))
        Jold = Jnew
        Jnew = (diff.transpose()[0].dot(diff.transpose()[0]))/(2*m)
        J.append(Jold)
    return T, Th, J


def Jay(T0, T1, X, Y):
    T = np.zeros((1, len(X[0])))
    T[0][0] = T0
    T[0][1] = T1
    Tx = np.dot(X, T.transpose())
    diff = (Y - Tx)
    temp = np.dot(diff.transpose()[0], (diff.transpose()[0]))/(2*len(X))
    return temp


X1, X2, xtemp, x = parseInput('Data/q1x.dat', True)
Y1, Y2, ytemp, y = parseInput('Data/q1y.dat', False)

T, Th, J = gradientDescent(X1, Y1, 0.02, 0.00000000001)
T2, Th2, J2 = gradientDescent(X2, Y2, 0.02, 0.00000000001)
print(T)
print(T2)

# plotPts(x,y)
# plotLine(T,X,x)
# mp.show()

fig = mp.figure()
ax = fig.gca(projection='3d')

X = np.arange(-2, 5, 0.25)
Y = np.arange(-2, 5, 0.25)
X, Y = np.meshgrid(X, Y)

Z = np.ones((len(X), len(Y)))
for i in xrange(len(X)):
    for j in xrange(len(Y)):
        Z[i][j] = Jay(X[i][j], Y[i][j], X1, Y1)

surface = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                          linewidth=0, antialiased=False)

# Contours

fig3 = mp.figure()
ax2 = fig3.add_subplot(111, projection='3d')
cset = ax2.contour(X, Y, Z, cmap=cm.coolwarm)
ax2.clabel(cset, fontsize=9, inline=1)

# Animation

for i in xrange(len(Th)):
    ax.scatter(Th[i][0], Th[i][1], J[i])
    ax2.scatter(Th[i][0], Th[i][1], J[i])
    mp.pause(0.0001)
