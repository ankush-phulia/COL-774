import math
import numpy as np

tr_file = 'Data/connect_4/train.data'
tst_file = 'Data/connect_4/test.data'


def getData(train_cases):
    with open(train_cases) as f:
        temp = f.readlines()
    m = len(temp)
    n = len(temp[0].split(','))
    X = np.zeros((m, n - 1))
    Y = np.zeros((m, 3), dtype=int)
    for i in xrange(m):
        temp2 = temp[i].split(',')
        X[i] = np.array(temp2[:-1])
        if temp2[n - 1] == 'win\n':
            Y[i] = np.array([1, 0, 0])
        elif temp2[n - 1] == 'draw\n':
            Y[i] = np.array([0, 1, 0])
        else:
            Y[i] = np.array([0, 0, 1])
    #X = np.insert(X, 0, 1, axis = 1)
    return X, Y


def g(x):
    return 1.0/(1.0 + np.exp(-x))
    # return np.log(1.0 + np.exp(x))


def g_(x):
    return (x * (1 - x))
    # return 1.0/(1.0 + np.exp(-x))
    # return (x > 0)


def denseOut(W, B, x):
    out = [None for _ in range(len(W))]
    out[0] = g(np.dot(x, W[0]) + B[0])
    for i in xrange(1, len(W)):
        out[i] = g(np.dot(out[i - 1], W[i]) + B[i])
    #temp = np.exp(out[-1])
    #cprob = temp/np.sum(temp, 1, None, None, True)
    return out


def backProp(W, B, out, Y, X, l):
    h = len(out)
    err = [None for _ in range(h)]
    update = [None for _ in range(h)]
    reg = l * (np.square(W[0]).sum() + np.square(W[1]).sum())
    diff = Y - out[-1] + reg
    # print(diff)
    #print(g_(out[h - 1]))
    err[-1] = diff * g_(out[-1])
    # print(err[h-1])
    for i in xrange(h - 2, -1, -1):
        err[i] = err[i + 1].dot(W[i + 1].transpose()) * g_(out[i])
        #update[i] = (np.dot(out[i].transpose(),err[i + 1]))
    update[0] = np.dot(X.transpose(), err[0])
    for i in xrange(1, h):
        update[i] = (np.dot(out[i - 1].transpose(), err[i]))
    return err, update


def train(X, Y, H=100, a=0.1, itern=500000):
    mx, my = len(X), len(Y)
    nx, ny = len(X[0]), 3
    W = [(2 * np.random.random((nx, H)) - 1)/100,
         (2 * np.random.random((H, ny)) - 1)/100]
    B = [np.zeros((1, H)), np.zeros((1, ny))]
    Xtemp = np.zeros((1, nx))
    Ytemp = np.zeros((1, ny))
    for i in xrange(itern):
        Xtemp[0] = X[i % mx]
        Ytemp[0] = Y[i % my]
        out = denseOut(W, B, Xtemp)
        err, upd = backProp(W, B, out, Ytemp, Xtemp, 0.00001)
        # print(upd)
        for i in xrange(len(W)):
            W[i] += a * (upd[i])
            B[i] += a * (err[i])
            # W[i] += a * (upd[i]) / math.sqrt(itern/len(X) + 1)
            # B[i] += a * (err[i]) / math.sqrt(itern/len(X) + 1)
    return W, B


def test(W, B, X, Y):
    correct = 0
    confusion = np.zeros((3, 3), dtype=int)
    for i in xrange(len(X)):
        a = np.argmax(denseOut(W, B, X[i:i+1])[1])
        b = np.argmax(Y[i])
        confusion[a][b] += 1
        if a == b:
            correct += 1
    print(confusion)
    print (correct * 100.0 / len(X))


X, Y = getData(tr_file)
Xt, Yt = getData(tst_file)
# print(X[:1])
# print(Y[:1])
print('Training')
W, B = train(X, Y, 100)
print('Training Data')
test(W, B, X, Y)
print('Testing Data')
test(W, B, Xt, Yt)
