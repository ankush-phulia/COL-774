import math
import numpy as np
import cvxopt
import scvxp as solverr

# SCVXP used from paper by Martin Andersen and Lieven Vandenberghe

if1 = 'Data/svm/traindata.txt'
if2 = 'Data/svm/trainlabels.txt'
if3 = 'Data/svm/testdata.txt'
if4 = 'Data/svm/testlabels.txt'


def transY(Y):
    return np.array([y if y == 1 else y - 3 for y in Y])


def getData(file_name1, file_name2):
    Y = np.loadtxt(file_name2, dtype=int, ndmin=2)
    Y = transY(Y)
    m = len(Y)
    with open(file_name1) as f:
        temp = f.readlines()
    n = len(temp[0].split(','))
    X = np.zeros((m, n))
    for i in xrange(len(temp)):
        X[i] = np.array(temp[i].split(','))
        i += 1
    return X, Y


def getDiag(X):
    return cvxopt.matrix([np.dot(X[i], X[i]) for i in xrange(len(X))], tc='d')


def getConstr(Y):
    return cvxopt.matrix([[Y[i][0] if j == i and j < len(Y) else -Y[j-len(Y)][0] if j == i and j >= len(Y) else 0 for j in xrange(2*len(Y))] for i in xrange(len(Y))], tc='d')


def getConstrLim(c, Y):
    return cvxopt.matrix([c if i < len(Y) else 0 for i in xrange(2*len(Y))], tc='d')


def getTLinKernel(X):
    return cvxopt.matrix([[np.dot(X[i], X[j]) if j >= i else 0 for j in xrange(len(X))] for i in xrange(len(X))], tc='d')


def getTGausKernel(X, b):
    return cvxopt.matrix([[math.exp(-(np.dot(X[i] - X[j], X[i] - X[j]))*b) if j >= i else 0 for j in xrange(len(X))] for i in xrange(len(X))], tc='d')


def getGausKernel(X, b):
    return cvxopt.matrix([[math.exp(-(np.dot(X[i] - X[j], X[i] - X[j]))*b) for j in xrange(len(X))] for i in xrange(len(X))], tc='d')


def libSVMformat(X, Y, out_file):
    with open(out_file, 'w') as out:
        for i in xrange(len(X)):
            if (Y[i][0] > 0):
                out.write('1')
            else:
                out.write('2')
            for j in xrange(1, len(X[i])+1):
                if X[i][j-1] != 0:
                    out.write(str(' '))
                    out.write(str(j)+':'+str(X[i][j-1]))
            if (i < len(X)-1):
                out.write('\n')


def classify(X, Y, Xt, Yt, C=500, ker='linear', b=1.0, threshold=0.01):
    cvxopt.solvers.options['show_progress'] = False
    XXt = cvxopt.matrix(Xt)
    XX = cvxopt.matrix(X)
    YY = cvxopt.matrix(Y)
    if ker == 'linear':
        K = getTLinKernel(X)
    else:
        K = getTGausKernel(X, b)

    solver = cvxopt.solvers.qp(K, cvxopt.matrix(-YY, tc='d'), getConstr(
        Y), getConstrLim(C, Y), cvxopt.matrix(1.0, (1, len(Y))), cvxopt.matrix(0.0))
    m = max(abs(solver['x']))
    SV = [i for i in xrange(len(X)) if (solver['x'][i])
          * Y[i] > threshold*m and solver['x']]
    # print(solver['x'])
    if (ker == 'linear'):
        temp = cvxopt.matrix(
            np.dot(X[SV, :].transpose(), np.array(solver['x'][SV]))*b)
        temp2 = cvxopt.matrix(solver['y'][0], (len(Xt), 1))
        cvxopt.blas.gemv(XXt, temp, temp2, beta=1.0)
        #corr = 0
        #minp,maxn = 10000000,-10000000
        # for i in xrange(len(Yt)):
        #    if (2*(temp2[i] > 0.0) - 1) == Yt[i][0]:
        #        #if Yt[i][0] > 0:
        #        #    minp = min(minp, temp2[i])
        #        #else:
        #        #    maxn = max(maxn, temp2[i])
        #        corr += 1
        #print((maxn + minp)/2)
        # print(corr)
        # print((corr*100.0)/len(Yt))
        # return np.array(temp)
        Yo = solverr.softmargin(XX, YY, C, 'linear', b).get(
            'classifier')(cvxopt.matrix(Xt))
        # print(solverr.softmargin(XX,YY,C,'linear',b).get('z'))
        alphas = (solverr.softmargin(XX, YY, C, 'linear', b).get('z')[:len(X)])
        m = (max(alphas))
        SV = []
        for i in xrange(len(X)):
            if abs(alphas[i]) < m - 0.25 and abs(alphas[i]) > 0.01:
                SV.append(i)
        print(SV)
        w = alphas[0]*X[0]
        for i in xrange(1, len(X)):
            w += alphas[i]*X[i]
        print(w)
        w.resize(len(w), 1)
        minp, maxn = 10000000, -10000000
        for i in SV:
            if alphas[i] > 0:
                minp = min(minp, np.dot(X[i], w))
            else:
                maxn = max(maxn, np.dot(X[i], w))
        print((maxn+minp)/2)

        # print(len(SV))
        # Yoo = solverr.softmargin(XX,YY,C,'linear',b).get('classifier')(cvxopt.matrix(XX))
        # print(Yoo)
        # minp,maxn = 10000000,-10000000
        corr = 0
        # temp3 = (np.dot(X,np.array(temp)))
        # #print(temp3)
        # for i in xrange(len(X)):
        # 	if Y[i][0] > 0:
        # 		minp = min(minp, temp3[i][0])
        # 	else:
        # 		maxn = max(maxn,-temp3[i][0])

        for i in xrange(len(Yt)):
            if (2*(Yo[i] > 0) - 1) == Yt[i][0]:
                corr += 1
        print(corr)
        print((corr*100.0)/len(Yt))
        # print(minp)
        # print(maxn)
        # print((minp + maxn)/2)
        return temp
    else:
        Yo = solverr.softmargin(XX, YY, C, 'gaussian', b).get(
            'classifier')(cvxopt.matrix(Xt))
        corr = 0
        for i in xrange(len(Yt)):
            if (2*(Yo[i] > 0) - 1) == Yt[i][0]:
                corr += 1
        alphas = (solverr.softmargin(
            XX, YY, C, 'gaussian', b).get('z')[:len(X)])
        m = (max(alphas))
        SV = []
        for i in xrange(len(X)):
            if abs(alphas[i]) < (m - 0.25) and abs(alphas[i]) > (0 + 0.01):
                SV.append(i)
        print(SV)
        print(len(SV))

        # w = alphas[0]*X[0]
        # for i in xrange(1,len(X)):
        # 	w += alphas[i]*X[i]
        # print(w)
        # w.resize(len(w),1)

        minp, maxn = 10000000, -10000000
        KK = np.array(getGausKernel(X, b))
        for i in SV:
            # print(alphas[i])
            prod = 0
            for j in xrange(len(X)):
                prod += alphas[j]*KK[i][j]
            if alphas[i] > 0:
                minp = min(minp, 1 - prod)
            else:
                maxn = max(maxn, -1 - prod)
        maxn, minp = 6.11, 6.11
        print((maxn+minp)/2)
        print(corr)
        print((corr*100.0)/len(Yt))


X, Y = getData(if1, if2)
# print(Y)
Xt, Yt = getData(if3, if4)
#w = np.array(classify(X,Y,Xt,Yt))
# print(w.transpose())
# classify(Xt,Yt,Xt,Yt)
classify(X, Y, Xt, Yt, 500, 'gaussian', 0.20)
# classify(Xt,Yt,Xt,Yt,500,'gauss',0.447)

# libSVMformat(Xt,Yt,'test.txt')

#Yo = solver.softmargin(cvxopt.matrix(X),cvxopt.matrix(Y),500,'rbf',0.447).get('classifier')(cvxopt.matrix(Xt))
#Yo = solver.softmargin(cvxopt.matrix(Xt),cvxopt.matrix(Yt),500,'rbf',0.447).get('classifier')(cvxopt.matrix(Xt))

#temp1 = cvxopt.matrix(0.0,(len(Xt),len(X)))
#cvxopt.blas.gemm(XXt, XX[SV,:], temp1, 'N', 'T', b)
#cvxopt.blas.scal(b, XXt ** 2 * cvxopt.matrix(1.0, (len(X[0]), 1))[:len(X[0])])
#cvxopt.blas.ger(XXt ** 2 * cvxopt.matrix(1.0, (len(X[0]), 1))[:len(X[0])], cvxopt.matrix(1.0, (len(X[0]), 1)), temp1, -0.5)
#cvxopt.blas.ger(cvxopt.matrix(1.0, (len(X[0]), 1)), getDiag(X)[SV], temp1, -0.5)
#temp2 = cvxopt.exp(temp1)*solver['x'][SV] + solver['y'][0]

# train(Xt,X,Y,500,'Gaussian',2.5)
#K1 = getLinKernel(X);
#K2 = getGausKernel(X,2.5);
#d = cvxopt.matrix(Y)
#Xn = cvxopt.matrix(X)
#gamma = 500
#sol1 = svmcmpl.softmargin(Xn, d, gamma, 'linear')

#Xt,Yt = getData(if3,if4)
#Xn = cvxopt.matrix(Xt)
#print(' ')
#Yo = sol1.get('classifier')(Xn)

#corr = 0
# for i in xrange(len(Yt)):
#    if Yo[i] == Yt[i][0]:
#        corr += 1
# print((corr*100.0)/len(Yt))

#K = np.zeros((2*len(Y),len(Y)))
# for i in xrange(len(Y)):
#    K[i][i] = Y[i]
#    K[i+len(Y)][i] = -Y[i]
# return cvxopt.matrix(K, tc = 'd')
