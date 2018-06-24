import numpy as np
from numpy import linalg
from numpy import random
np.set_printoptions(threshold=np.nan)

attr_file = 'Data/kmeans_data/attr.txt'
label_file = 'Data/kmeans_data/label.txt'


def getData(attributes, labels):
    X = np.loadtxt(attributes)
    Y = np.loadtxt(labels, dtype=int)
    # Y.resize((len(Y), 1))
    return X, Y


def getAccuracy(C, Y, k=6):
    labels = [np.bincount((np.array(Y[C == j]))).argmax() for j in xrange(k)]
    correct = [sum(Y[C == j] == labels[j]) for j in xrange(k)]
    return sum(correct) * 100.0 / len(Y)


def kMeans(X, k=6, itern=60, e=0):
    # initialise random centroids
    m, n = len(X), len(X[0])
    U = X[random.choice(m, k), :]
    Jold, Jnew = 2000000, 1000000
    Accuracy = 0.0
    while (Jold - Jnew > e):
        Jold = Jnew
        C = np.array([np.argmin([linalg.norm((X[i] - U[j]))
                                 for j in xrange(k)]) for i in xrange(m)])
        Jnew = sum([linalg.norm(X[i] - U[C[i]])**2 for i in xrange(m)])
        # print(Jnew)
        Accuracy = getAccuracy(C, Y)
        # print(Accuracy)
        U = [X[C == j].mean(axis=0) for j in xrange(k)]
        print(Jnew)
    # print(Accuracy)
    return np.array(U), C, Jnew, Accuracy


def kMeansRe(X, k=6, itern=60, e=0, restarts=0):
    print('Starting')
    Uo, Co, Jo, Ao = kMeans(X, k, itern, e)
    for r in xrange(restarts):
        print('Starting')
        U, C, J, A = kMeans(X, k, itern, e)
        print [len(X[C == j]) for j in xrange(k)]
        print J
        print A
        if J < Jo:
            Uo, Co, Jo, Ao = U, C, J, A
    print(' ')
    return Uo, Co, Jo, Ao


X, Y = getData(attr_file, label_file)
U, C, J, A = kMeansRe(X, restarts=20)
print('Least Error Found : ')
print(J)
print('Corresponding Accuracy : ')
print(A)
