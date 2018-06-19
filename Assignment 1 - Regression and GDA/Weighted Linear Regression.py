import math
import numpy as np
import matplotlib.pyplot as mp
from numpy.linalg import inv


def parseInput(in_file_name, chk):
    temp = (np.loadtxt(in_file_name, dtype=float, ndmin=2))
    if not(chk):
        return temp, temp.reshape((1, len(temp)))
    else:
        X = np.ones((len(temp), 2))
        X[:, 1:] = temp
        return X, temp.reshape((1, len(temp)))


def calcTheta(X, Y):
    Xt = np.transpose(X)
    temp = inv(np.dot(Xt, X))
    return np.dot(temp, (np.dot(Xt, Y)))


def calcTheta(X, W, Y):
    Xt = np.transpose(X)
    temp = inv(np.dot(Xt, np.dot(W, X)))
    return np.dot(temp, (np.dot(Xt, np.dot(W, Y))))


def calcW(x, X, t):
    m = len(X)
    W = np.zeros((m, m))
    for i in xrange(m):
        W[i][i] = math.exp(-((x - X[i][1])**2)/(2*t**2))
    return W


def plotPts(x, y):
    mp.plot(x[0], y[0], 'o')
    axes = mp.gca()
    axes.set_xlim([min(x[0]) - 1, max(x[0]) + 1])


def plotLine(T, X, x):
    Y = [np.dot(T[0], xi) for xi in X]
    mp.plot(x[0], Y)


def plotWeighted(X, Y, t):
    y = []
    for xi in X:
        W = calcW(xi[1], X, t)
        T = calcTheta(X, W, Y)
        xi.resize((1, 2))
        y.append(np.dot(xi, T)[0][0])
    return y


def Run():
    X, x = parseInput('Data/q3x.dat', True)
    Y, y = parseInput('Data/q3y.dat', False)

    plotPts(x, y)
    mp.plot(x[0], plotWeighted(X, Y, 10), 'o')
    mp.show()


if __name__ == '__main__':
    Run()
