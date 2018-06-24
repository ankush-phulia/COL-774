import numpy as np
import re
import time
import math
from sklearn import preprocessing, svm, tree, ensemble, naive_bayes
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score

train_file = 'Data/train.csv'
test_file = 'Data/test.csv'
num_attr = 4125


def getData(filename, flag):
    attributes_present = np.zeros(num_attr)
    X, Y = [], []
    with open(filename) as f:
        for x in f.readlines():
            temp = re.split(' |[:]', x.strip())
            if (flag == 1):
                Y.append(int(temp[0]))
            attr = np.zeros(num_attr)
            for i in xrange(flag, len(temp), 2):
                attr[int(temp[i])] = float(temp[i + 1])
                attributes_present[int(temp[i])] += 1
            X.append(attr)
    return np.array(X), Y, attributes_present


def getData1(filename, flag):
    attributes_present = np.zeros(num_attr)
    X, Y, I = [], [], []
    with open(filename) as f:
        c = 0
        for x in f.readlines():
            temp = re.split(' |[:]', x.strip())
            if (len(temp) < 8):
                if (flag == 1):
                    Y.append(int(temp[0]))
                attr = np.zeros(num_attr)
                for i in xrange(flag, len(temp), 2):
                    attr[int(temp[i])] = float(temp[i + 1])
                    attributes_present[int(temp[i])] += 1
                X.append(attr)
                I.append(c)
            c += 1
    return np.array(X), Y, attributes_present, I


def getData2(filename, flag):
    attributes_present = np.zeros(num_attr)
    X, Y, I = [], [], []
    with open(filename) as f:
        c = 0
        for x in f.readlines():
            temp = re.split(' |[:]', x.strip())
            if len(temp) > 7:
                if (flag == 1):
                    Y.append(int(temp[0]))
                attr = np.zeros(num_attr)
                for i in xrange(flag, len(temp), 2):
                    attr[int(temp[i])] = float(temp[i + 1])
                    attributes_present[int(temp[i])] += 1
                X.append(attr)
                I.append(c)
            c += 1
    return np.array(X), Y, attributes_present, I


def reduceData(X, attributes_present):
    useless = []
    for i in xrange(num_attr):
        if attributes_present[i] == 0:
            useless.append(i)
    X = np.delete(X, useless, axis=1)
    # pca = PCA(n_components = 23, svd_solver = 'full')
    # X = pca.fit_transform(X)
    return X


def Cross(X, Y, typ=0, n=5):
    if typ == 0:
        print('Starting SVM')
        classifier = make_pipeline(
            preprocessing.StandardScaler(), svm.LinearSVC())
    elif typ == 1:
        print('Starting DTree')
        classifier = make_pipeline(
            preprocessing.StandardScaler(), tree.DecisionTreeClassifier())
    elif typ == 2:
        print('Starting RForest')
        classifier = make_pipeline(
            preprocessing.StandardScaler(), ensemble.RandomForestClassifier())
    elif typ == 3:
        print('Starting GaussianNB')
        classifier = make_pipeline(
            preprocessing.StandardScaler(), naive_bayes.GaussianNB())
    elif typ == 4:
        print('Starting AdaBoost')
        classifier = make_pipeline(
            preprocessing.StandardScaler(), ensemble.AdaBoostClassifier())
    elif typ == 5:
        print('Starting Bagging')
        classifier = make_pipeline(
            preprocessing.StandardScaler(), ensemble.BaggingClassifier())
    elif typ == 6:
        print('Starting ExTree')
        classifier = make_pipeline(
            preprocessing.StandardScaler(), ensemble.ExtraTreesClassifier())
    elif typ == 7:
        print('Starting GradBoost')
        classifier = make_pipeline(
            preprocessing.StandardScaler(), ensemble.GradientBoostingClassifier())
    else:
        return
    scores = cross_val_score(classifier, X, Y, cv=n)
    # print scores
    print sum(scores)/n


def ExTree(X, Y, nest, mf, md, mss, msl, mwfl, mln, mis, bs):
    print('Extree')
    classifier = make_pipeline(preprocessing.StandardScaler(), ensemble.RandomForestClassifier(
        n_estimators=nest, max_features=mf, max_depth=md, min_samples_split=mss,
        min_samples_leaf=msl, min_weight_fraction_leaf=mwfl, max_leaf_nodes=mln,
        min_impurity_split=mis, bootstrap=bs))
    # print(classifier.get_params())
    scores = cross_val_score(classifier, X, Y, cv=5)
    print sum(scores)/5


def test(X, Y, Xt, filename):
    model = make_pipeline(preprocessing.StandardScaler(),
                          ensemble.ExtraTreesClassifier(n_estimators=50))
    # model = ensemble.ExtraTreesClassifier()
    # print('Fitting')
    model.fit(X, Y)
    # print('Predicting')
    pred = model.predict(Xt)
    # with open(filename, 'w+') as f:
    print 'ID,TARGET'
    # f.write('ID,TARGET\n')
    for i in xrange(len(Xt)):
        print i, pred[i]
        # f.write(i,',',pred[i])


def test(X, Y, Xt, It, filename):
    model = make_pipeline(preprocessing.StandardScaler(
    ), ensemble.ExtraTreesClassifier(n_estimators=50, max_features=num_attr))
    # model = ensemble.ExtraTreesClassifier()
    # print('Fitting')
    model.fit(X, Y)
    # print('Predicting')
    pred = model.predict(Xt)
    # with open(filename, 'w+') as f:
    print 'ID,TARGET'
    # f.write('ID,TARGET\n')
    j = 0
    for i in xrange(len(Xt)):
        print It[j], pred[i]
        j += 1
        # f.write(i,',',pred[i])

# print('Collecting Training Data')
# X, Y, attributes_present, I = getData2(train_file, 1)
# print len(I)
# print('Collecting Testing Data')
# Xt, Yt, attr, It = getData2(test_file, 0)
# print len(It)

# X, Y, attributes_present, I = getData1(train_file, 1)
# print len(I)
# print('Collecting Testing Data')
# Xt, Yt, attr, It = getData1(test_file, 0)
# print len(It)


# print('Reducing Data')
X = reduceData(X, attributes_present)
Xt = reduceData(Xt, attributes_present)

# ExTree(X, Y, 10, int(math.sqrt(num_attr)), None, 2, 1, 0., None, 1e-7)
# ExTree(X, Y, 10, 0.9, None, 2, 1, 0., None, 1e-7)
# ExTree(X, Y, 10, int(math.log(231,2)), None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 10, int(math.sqrt(231)), None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 10, int(math.log(4125,2)), None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 50, 50, None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 50, 50, None, 2, 1, 0., None, 1e-7, True)
# ExTree(X, Y, 10, 45, None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 10, 40, None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 10, 35, None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 10, 30, None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 10, 25, None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 10, 20, None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 10, 75, None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 10, 70, None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 10, 60, None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 10, 55, None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 10, 64, None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 30, int(math.sqrt(num_attr)), None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 30, int(math.sqrt(num_attr)), None, 2, 1, 0., None, 1e-7, True)
# ExTree(X, Y, 50, int(math.sqrt(num_attr)), None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 50, int(math.sqrt(num_attr)), None, 2, 1, 0., None, 1e-7, True)
# ExTree(X, Y, 70, int(math.sqrt(num_attr)), None, 2, 1, 0., None, 1e-7, False)
# ExTree(X, Y, 70, int(math.sqrt(num_attr)), None, 2, 1, 0., None, 1e-7, True)
# ExTree(X, Y, 10, 0.6, None, 2, 1, 0., None, 1e-7)
# ExTree(X, Y, 10, 0.8, None, 2, 1, 0., None, 1e-7)
# ExTree(X, Y, 10, None, None, 2, 1, 0., None, 1e-7)
# ExTree(X, Y, 30, None, None, 2, 1, 0., None, 1e-7)
# ExTree(X, Y, 50, None, None, 2, 1, 0., None, 1e-7)
# ExTree(X, Y, 50, None, None, 2, 1, 0., None, 1e-7)
# ExTree(X, Y, 60, None, None, 2, 1, 0., None, 1e-7)
# ExTree(X, Y, 60, None, None, 2, 1, 0., None, 1e-7)
# Cross(X, Y, 0, 5)
# Cross(X, Y, 1, 5)
# Cross(X, Y, 2, 5)
# Cross(X, Y, 3, 5)
# Cross(X, Y, 4, 5)
# Cross(X, Y, 5, 5)
# Cross(X, Y, 6, 5)
# Cross(X, Y, 7, 5)
# print('Testing')
test(X, Y, Xt, It, 'out.csv')
