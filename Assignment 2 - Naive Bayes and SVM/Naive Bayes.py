from collections import defaultdict
import math
import operator
import numpy as np

# stop_words = ['a','an','the','is','was','and','would',
#              'could','should','be','has','then','this','that',
#              'I','am','what','when','how','had']
stop_words = []


def add_to_dict(d, l):
    for i in xrange(1, len(l)):
        d[l[i]] += 1


def add_to_ddict(an, d, l):
    an[l[0]] += 1
    for i in xrange(1, len(l)):
        d[l[0]][l[i]] += 1


def remStopWords(l, stopw):
    return [item for item in l if item not in stopw]


def trainCases(in_file_name):
    total_articles = 0
    art_nums = defaultdict(int)
    class_vocab = defaultdict(lambda: defaultdict(int))
    vocab = defaultdict(int)
    with open(in_file_name) as f:
        for line in f:
            total_articles += 1
            words = remStopWords(line.split(), stop_words)
            add_to_dict(vocab, words)
            add_to_ddict(art_nums, class_vocab, words)
    return total_articles, art_nums, class_vocab, vocab


def testCases(in_file_name):
    tc = []
    with open(in_file_name) as f:
        for line in f:
            wordmap = defaultdict(int)
            words = remStopWords(line.split(), stop_words)
            add_to_dict(wordmap, words)
            tc.append((words[0], wordmap))
    return tc


def prob(testMap, classMap, vocab, Anum):
    wordsNum = sum(classMap.values())
    ans = 1.0
    for word, count in testMap.items():
        y = wordsNum + len(vocab)
        ans += ((math.log(classMap[word] + 1.0) - math.log(y))*count)
    return ans


def classify(TA, ANum, CV, V, TestCases):
    confusion = defaultdict(lambda: defaultdict(int))
    correct = 0
    for tc, dtc in TestCases:
        ps = np.zeros(len(ANum))
        i = 0
        for group, gvocab in CV.items():
            ps[i] = prob(dtc, gvocab, V, ANum) + \
                math.log(ANum[group]) - math.log(TA)
            i += 1
        pred = np.where(ps == np.amax(ps))
        confusion[tc][list(CV.keys())[pred[0][0]]] += 1
        if tc == list(CV.keys())[pred[0][0]]:
            correct += 1
    return confusion, (correct*100.0)/len(TestCases)


def Run():
    TA, ANum, CV, V = trainCases('Data/r8-train-all-terms-new.txt')
    TestCases = testCases('Data/r8-test-all-terms-new.txt')

    CMap, acc = classify(TA, ANum, CV, V, TestCases)
    print(acc)
    #MaxArt = max(ANum.items(), key=(lambda key : ANum[key]))[0]
    #randomAcc = 200.0/len(ANum)
    #majAcc = sum(CMap[MaxArt].values())*100.0/len(TestCases)
    #maxDiag,maxDiagKey = 0,0
    # for key in ANum.keys():
    #    if CMap[key][key] > maxDiag:
    #        maxDiagKey,maxDiag = key, CMap[key][key]

    # print(ANum)
    # print(len(ANum)/2)
    # print(randomAcc)
    # print(majAcc)
    # print(maxDiag)
    # print(maxDiagKey)
    # print(acc)
    # print(MaxArt)
    print(CMap)


if __name__ == '__main__':
    Run()
