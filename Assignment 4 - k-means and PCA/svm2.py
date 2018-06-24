import os
import time
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from PIL import Image
from scipy.misc import imread, toimage

dir1 = 'Data/lfw_easy'  # colored -> greyscale
dir2 = 'Data/orl_faces'  # greyscale


def getData1(folder):
    X, Y, ycurr = [], [], 1
    for sub_folder in os.listdir(folder):
        if not(sub_folder.endswith(".txt")):
            sub_folder = os.path.join(folder, sub_folder)
            for file in os.listdir(sub_folder):
                X.append(imread(os.path.join(sub_folder, file)).flatten())
                Y.append(ycurr)
            ycurr += 1
        else:
            continue
    X = np.array(X)
    M = X.mean(axis=0)
    X = X - M
    return X, Y


def crossVal(X, Y, n=10):
    mcsvm = svm.SVC()
    start = time.time()
    scores = cross_val_score(mcsvm, X, Y, cv=n)
    end = time.time()
    print scores
    print sum(scores)/n
    print end - start


X1, Y1 = getData1(dir1)
X2, Y2 = getData1(dir2)
print('Data Set 1 Cross validation Scores before PCA')
X1 /= X1.std(axis=0)
crossVal(X1, Y1)
print('Data Set 2 Cross validation Scores before PCA')
X2 /= X2.std(axis=0)
crossVal(X2, Y2)
# X1 = np.loadtxt('Projected1.txt')
# X2 = np.loadtxt('Projected2.txt')
# print('Data Set 1 Cross validation Scores after PCA')
# crossVal(X1, Y1)
# print('Data Set 2 Cross validation Scores after PCA')
# crossVal(X2, Y2)

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

# def getData2(folder):
# 	X, Y = [], []
# 	for sub_folder in os.listdir(folder):
# 		if not(sub_folder.endswith(".txt")):
# 			sub_folder = os.path.join(folder, sub_folder)
# 			for file in os.listdir(sub_folder):
# 				X.append(imread(os.path.join(sub_folder, file)).flatten())
# 				Y.append(i)
# 		else:
# 			continue
# 	return np.array(X), Y

# print(Y1)
# print(Y2)
# print(Y1)
# print(X.shape)
# print(X[0])
# X = X.reshape((len(X),(len(X[0])*len(X[0][0]))))
# print(X.shape)
# print(X[0])
# def prepCrossSet(X, Y, n = 10):
# 	chunk = len(X) / n
# 	Xc = [X[i * chunk:(i + 1) * chunk] for i in xrange(n)]
# 	Yc = [Y[i * chunk:(i + 1) * chunk] for i in xrange(n)]
# 	return Xc, Yc

# def conc(L, k):
# 	n = len(L)
# 	LL = L[(k + 1)%n]
# 	for i in xrange(len(L)):
# 		if (i != k and i != (k + 1) % n):
# 			np.append(LL, L[i], axis = 0)
# 	return LL
