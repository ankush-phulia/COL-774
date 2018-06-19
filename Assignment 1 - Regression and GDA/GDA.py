import math
import numpy as np
import matplotlib.pyplot as mp
from numpy.linalg import inv
from numpy.linalg import det


def convert(s):
    return int(s == 'Alaska')


def parseInput(in_file_name, chk):
    if chk:
        temp = (np.loadtxt(in_file_name, dtype=str, ndmin=2))
        return map(convert, temp)
    return (np.loadtxt(in_file_name, dtype=float, ndmin=2))


def get_params(X, Y):
    u0 = np.zeros(len(X[0]))
    n0 = 0
    u1 = np.zeros(len(X[0]))
    n1 = 0
    s = np.zeros((len(X[0]), len(X[0])))
    s0 = np.zeros((len(X[0]), len(X[0])))
    s1 = np.zeros((len(X[0]), len(X[0])))
    for i in xrange(len(X)):
        xi = X[i]
        yi = Y[i]
        if yi == 0:
            u0 += xi
            n0 += 1
        else:
            u1 += xi
            n1 += 1
    u0 /= n0
    u1 /= n1
    for i in xrange(len(X)):
        xi = X[i]
        yi = Y[i]
        if yi == 0:
            t1 = (xi - u0)
            t1.resize(1, len(xi))
            temp = np.dot(t1.transpose(), t1)
            s0 += temp
        else:
            t1 = (xi - u1)
            t1.resize(1, len(xi))
            temp = np.dot(t1.transpose(), t1)
            s1 += temp
        s += temp
    return n0, n1, u0, u1, s0/n0, s1/n1, s/(n0 + n1)


def computeBdry1(p1, p2, u0, u1, s, X):
    ss = inv(s)
    temps = (np.dot(ss, (u0-u1)))
    tempp = math.log(p1/p2)
    tempu0 = 0.5*np.dot(np.dot(u0.transpose(), ss), u0)
    tempu1 = 0.5*np.dot(np.dot(u1.transpose(), ss), u1)
    temp = tempp + tempu0 - tempu1
    x1, x2 = [], []
    for xi in X:
        x1.append(xi[0])
        x2.append((temp - xi[0]*temps[0])/temps[1])
    return x1, x2


def computeBdry2(p1, p2, u0, u1, s0, s1, X):
    ss0 = inv(s0)
    ss1 = inv(s1)
    temp1 = ss0 - ss1
    temp2 = (-2)*(np.dot(ss0, u0) - np.dot(ss1, u1))
    temp3 = np.dot(np.dot(u0.transpose(), ss0), u0) - \
        np.dot(np.dot(u1.transpose(), ss1), u1)
    temp4 = math.log(math.sqrt(abs(det(ss0) / det(ss1))))
    a = temp1[0][0]
    b = temp1[1][0] + temp1[0][1]
    c = temp1[1][1]
    d = temp2[0]
    e = temp2[1]
    f = temp3 + temp4
    x1, x2 = [], []
    for xi in X:
        x1.append(xi[0])
        temp = -(e + b*xi[0]) - math.sqrt((e + b*xi[0])
                                          ** 2 - 4*c*(a*xi[0]**2 + d*xi[0] + f))
        x2.append(temp/(2*c))
    return x1, x2


def Run():
    X = parseInput('Data/q4x.dat', False)
    Y = parseInput('Data/q4y.dat', True)

    X0 = np.array([X[i] for i in xrange(len(X)) if Y[i] == 0])
    X1 = np.array([X[i] for i in xrange(len(X)) if Y[i] == 1])
    X0t = X0.transpose()
    X1t = X1.transpose()
    mp.plot(X0t[0], X0t[1], 'o', color='red')
    mp.plot(X1t[0], X1t[1], 'o', color='green')

    p1, p2, u0, u1, s0, s1, s = get_params(X, Y)
    # print(get_params(X,Y))

    x, y = computeBdry2(p1, p2, u0, u1, s0, s1, X)
    mp.plot(x, y, 'o')
    mp.show()


if __name__ == '__main__':
    Run()
