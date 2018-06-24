import os
import numpy as np
import sklearn as sk
from PIL import Image
from scipy.misc import imread, toimage


dir1 = 'Data/lfw_easy'  # colored
dir2 = 'Data/orl_faces'  # greyscale


def toGreyScale(folder):
    for sub_folder in os.listdir(folder):
        if not(sub_folder.endswith(".txt")):
            sub_folder = os.path.join(folder, sub_folder)
            for file in os.listdir(sub_folder):
                file = os.path.join(sub_folder, file)
                img = Image.open(file).convert('L')
                img.save(file)
        else:
            continue


def getData(folder):
    X, F = [], []
    for sub_folder in os.listdir(folder):
        if not(sub_folder.endswith(".txt")):
            sub_folder = os.path.join(folder, sub_folder)
            for file in os.listdir(sub_folder):
                X.append(imread(os.path.join(sub_folder, file)))
                F.append(os.path.join(sub_folder, file))
        else:
            continue
    return np.array(X), F


def print2Data(data, file):
    with open(file, 'w+') as f:
        for arr in data:
            for item in arr:
                f.write(str(item) + ' ')
            f.write('\n')


def getAverage(folder, output_file):
    toimage(getData(folder)[0].mean(axis=0),
            cmin=0.0, cmax=255).save(output_file)


def eigenFaces(folder, num_faces=50, display=3):
    X, F = getData(folder)
    n, r, c = X.shape
    X = X.reshape((n, r * c))
    M = X.mean(axis=0)
    Xmad = X - M
    U, C, V = np.linalg.svd(Xmad.T, full_matrices=False)
    minV, maxV = np.amin(U), np.amax(U)
    UscT = ((U - minV) * (255.0/(maxV - minV))).T
    for i in xrange(display):
        out_file = 'eigen_face_' + str(i + 1) + '.jpg'
        toimage(UscT[i].reshape(r, c)).save(out_file)
    W = np.dot(Xmad, U[:, :num_faces])
    # W /= W.std(axis = 0)
    return U.T[:num_faces], W, M, X, r, c


def project(x, w, M, PC):
    return w
    # return M + np.dot(w, PC)


def allProjs(X, W, M, PC, out_file):
    data = [project(X[i], W[i], M, PC) for i in xrange(len(X))]
    print2Data(data, out_file)
    return data


def compare_ind(i, X, F, Proj, r, c):
    outfile = 'Image' + str(i + 1)
    toimage(X[i].reshape(r, c)).save((outfile + 'O_id.jpg'))
    toimage(Proj[i].reshape(r, c)).save((outfile + 'P_id.jpg'))


def compare_name(fid, X, F, Proj, r, c):
    i = F.index(fid)
    outfile = 'Image' + str(i + 1)
    toimage(X[i].reshape(r, c)).save((outfile + 'O_name.jpg'))
    toimage(Proj[i].reshape(r, c)).save((outfile + 'P_name.jpg'))

# getAverage(dir1, 'out1.jpg')
# getAverage(dir2, 'out2.jpg')

# toGreyScale(dir1)


PC, W, M, X, r, c = eigenFaces(dir1)
# print2Data(W, 'PrincipalComp1.txt')
proj = allProjs(X, W, M, PC, 'Projected1.txt')

PC, W, M, X, r, c = eigenFaces(dir2)
# print2Data(W, 'PrincipalComp2.txt')
proj = allProjs(X, W, M, PC, 'Projected2.txt')

# X = getData(dir1)
# n, r, c = X.shape
# X.reshape((n, r * c))
# proj = np.loadtxt('Projected1.txt')
# compare(0, X, proj, r, c)

# X, F = getData(dir2)
# n, r, c = X.shape
# X.reshape((n, r * c))
# proj = np.loadtxt('Projected2.txt')
# compare_ind(0, X, F, proj, r, c)
# compare_name("Data/orl_faces/s21/1.pgm", X, F, proj, r, c)
# compare(99, X, proj, r, c)
# compare(199, X, proj, r, c)
# compare(299, X, proj, r, c)
# compare(399, X, proj, r, c)

# PC, W, M, X, r, c = eigenFaces(dir2, 50)
# proj = allProjs(X, W, M, PC, 'Projected2.txt')
# compare(0, X, proj, r, c)
