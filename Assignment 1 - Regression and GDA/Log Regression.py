import math
import numpy as np
import matplotlib.pyplot as mp
from numpy.linalg import inv


def parseInput(in_file_name, chk):
    temp = (np.loadtxt(in_file_name, dtype=float, ndmin=2))
    if chk == False:
        return temp, temp.reshape((1, len(temp)))
    else:
        X = np.ones((len(temp), 3))
        X[:, 1:] = temp
        return X, temp.reshape((2, len(temp)))


def logistic(x):
    return 1.0/(1.0 + math.exp(-x))


def Hessian(T, X):
    n = len(T)
    H = np.zeros((n, n))
    for i in xrange(n):
        for j in xrange(n):
            for xk in X:
                Tx = np.dot(T.transpose(), xk)
                g = logistic(Tx)
                H[i][j] += -g*(1-g)*xk[i]*xk[j]
    return H


def NewtonRaphson(X, Y, epsilon):
    T = np.zeros(len(X[0]))
    H = Hessian(T, X)
    Tx = np.dot(X, T.transpose())
    g = np.array(map(logistic, Tx))
    g.resize(len(Y), len(Y[0]))
    temp = (np.dot(inv(H), ((Y-g)*X).transpose()).transpose()).sum(axis=0)
    diff = np.dot(temp, temp)
    T -= temp
    while (diff > epsilon):
        Tx = np.dot(X, T.transpose())
        g = np.array(map(logistic, Tx))
        g.resize(len(Y), len(Y[0]))
        temp = (np.dot(inv(H), ((Y-g)*X).transpose()).transpose()).sum(axis=0)
        diff = np.dot(temp, temp)
        T -= temp
    return T


def eval(T, X):
    x1 = []
    x2 = []
    for xi in X:
        x2.append(-(T[0] + T[1]*xi[1])/T[2])
        x1.append(xi[1])
    return x1, x2


X, x = parseInput('Data/q2x.dat', True)
Y, y = parseInput('Data/q2y.dat', False)
T = NewtonRaphson(X, Y, 0.00000000001)
X0 = np.array([X[i] for i in xrange(len(X)) if y[0][i] == 0])
X1 = np.array([X[i] for i in xrange(len(X)) if y[0][i] == 1])
X0t = X0.transpose()
X1t = X1.transpose()
x1, x2 = eval(T, X)
print(T)
mp.plot(X0t[1], X0t[2], 'o', color='red')
mp.plot(X1t[1], X1t[2], 'o', color='green')
mp.plot(x1, x2)
mp.show()
