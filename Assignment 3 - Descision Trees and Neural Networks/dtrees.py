from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def getData(file_name):
    with open(file_name) as f:
        temp = f.readlines()
        X = [map(int, temp[i].strip('\n').split(' '))[:-1]
             for i in xrange(1, len(temp))]
        Y = [int(temp[i].strip('\n').split(' ')[-1])
             for i in xrange(1, len(temp))]
    return X, Y


def eval_tree(X, Y, Xt, Yt):
    dtree = DecisionTreeClassifier(
        min_samples_split=2, min_samples_leaf=1, max_depth=35)
    dtree.fit(X, Y)
    pred = dtree.predict(Xt)
    correct = 0
    for i in xrange(len(Xt)):
        if (pred[i] == Yt[i]):
            correct += 1
    return 100.0 * correct / len(Yt)


def eval_forest(X, Y, Xt, Yt):
    dtree = RandomForestClassifier(
        n_estimators=1, max_features=54, bootstrap=False)
    dtree.fit(X, Y)
    pred = dtree.predict(Xt)
    correct = 0
    for i in xrange(len(Xt)):
        if (pred[i] == Yt[i]):
            correct += 1
    return 100.0 * correct / len(Yt)


train_file = 'Data/covType/train.dat'
valid_file = 'Data/covType/valid.dat'
test_file = 'Data/covType/test.dat'

X, Y = getData(train_file)
Xv, Yv = getData(valid_file)
Xt, Yt = getData(test_file)

print('Evaluating Decision Trees')
print(eval_tree(X, Y, Xv, Yv))
print(eval_tree(X, Y, Xt, Yt))
# print(' ')
print('Evaluating Random Forests')
print(eval_forest(X, Y, Xv, Yv))
print(eval_forest(X, Y, Xt, Yt))
