import time
import numpy as np
from sklearn import svm, tree, ensemble
from sklearn.model_selection import cross_val_score

attr_file = 'Data/kmeans_data/attr.txt'
label_file = 'Data/kmeans_data/label.txt'


def getData(attributes, labels):
    X = np.loadtxt(attributes)
    Y = np.loadtxt(labels)
    return X, Y


def crossVal(X, Y, n=10):
    mcsvm = svm.SVC()
    start = time.time()
    scores = cross_val_score(mcsvm, X, Y, cv=n)
    end = time.time()
    print scores
    print sum(scores)/n
    print end - start


X, Y = getData(attr_file, label_file)
crossVal(X, Y)

# Xc, Yc = prepCrossSet(X, Y, n)
# for i in xrange(n):
# 	#get the set
# 	Xtr, Ytr = conc(Xc, i), conc(Yc, i)
# 	#fit
# 	mcsvm.fit(Xtr, Ytr)
# 	#predict
# 	pred = mcsvm.predict(Xc[i])
# 	correct = 0
# 	test = len(pred)
# 	#evaluate acc
# 	for j in xrange(test):
# 		if pred[j] == Yc[i][j]:
# 			correct += 1
# 	print(i)
# 	print(correct * 100.0 / test)


def prepCrossSet(X, Y, n = 10):
        chunk = len(X) / n
        Xc = [X[i * chunk:(i + 1) * chunk] for i in xrange(n)]
        Yc = [Y[i * chunk:(i + 1) * chunk] for i in xrange(n)]
        return Xc, Yc

def conc(L, k):
        n = len(L)
        LL = L[(k + 1)%n]
        for i in xrange(len(L)):
                if (i != k and i != (k + 1) % n):
                        np.append(LL, L[i], axis = 0)
        return LL
