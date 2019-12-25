# encoding=utf-8

import csv
import pickle
import re
from collections import Counter
from functools import partial
from math import fabs, log
from multiprocessing import Pool

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from bayes import BayesSpam
from rule import Rule, qqPredict


def calAccuracy(devs, outs):
    n = len(outs)
    equ = 0
    fake_true = 0
    fake_false = 0
    ftrue = open("logs/fake_true.txt", "w")
    ffalse = open("logs/fake_false.txt", "w")
    fclose = open("logs/close.txt", "w")
    # assert len(devs) == len(outs)
    for i in range(n):
        if int(devs[i][1]) == outs[i]:
            equ += 1
        elif devs[i][1] == '0':
            fake_true += 1
            print(devs[i][0], devs[i][1], outs[i], outs[i], file=ftrue)
        else:
            fake_false += 1
            print(devs[i][0], devs[i][1], outs[i], outs[i], file=ffalse)
        # if fabs(outs[i][1]) < 0.2:
        #     print(devs[i][0], devs[i][1], outs[i], outs[i], file=fclose)
    print("fake true = %f(%d), fake false = %f(%d)" %
          (fake_true/n, fake_true, fake_false/n, fake_false))
    return equ/n


def dump(rets):
    with open('answer.txt', 'w') as f:
        print("\n".join(map(str, map(lambda x: x, rets))), file=f)

def merge(lst):
    assert len(lst) > 0
    ret = np.array(lst[0])
    ret = ret.reshape(ret.shape[0], -1)
    for i in range(1, len(lst)):
        a = np.array(lst[i])
        ret = np.concatenate((ret, a.reshape(a.shape[0], -1)), axis=1)
    return ret

def main():
    with open("../data/train.csv", 'r', encoding='utf-8') as f:
        train_raw = list(csv.reader(f))

    with open("../data/dev.csv", 'r', encoding='utf-8') as f:
        dev_raw = list(csv.reader(f))

    with open("../data/test.csv", 'r', encoding='utf-8') as f:
        test_raw = list(csv.reader(f))[1:]

    bs = BayesSpam()

    print("_______bayes________")
    bs.train(train_raw)

    bayes_result_train = bs.predict(train_raw)
    bayes_result_dev = bs.predict(dev_raw)
    bayes_result_test = bs.predict(test_raw)
    print("test completed")

    devAccuracy = calAccuracy(dev_raw, bayes_result_dev)
    trainAccuracy = calAccuracy(train_raw, bayes_result_train)
    print("train acc =", trainAccuracy)
    print("dev acc =", devAccuracy)
    print("_______bayes________")

    print("_______ip________")
    rule = Rule()
    rule.ipTrain(train_raw)

    ip_result_train = rule.ipPredict(train_raw)
    ip_result_dev = rule.ipPredict(dev_raw)
    ip_result_test = rule.ipPredict(test_raw)
    print("_______ip________")

    with Pool() as pool:
        qq_result_train = pool.map(qqPredict, train_raw)
        qq_result_dev = pool.map(qqPredict, dev_raw)
        qq_result_test = pool.map(qqPredict, test_raw)

    print("______gdbt_______")
    fuse_train = merge([bayes_result_train, ip_result_train, qq_result_train])
    fuse_dev = merge([bayes_result_dev, ip_result_dev, qq_result_dev])
    fuse_test = merge([bayes_result_test, ip_result_test, qq_result_test])

    label_train = np.array(list(map(lambda x : int(x[1]), train_raw)))

    print(fuse_train)
    print(label_train)
    dt = DecisionTreeClassifier()
    dt.fit(fuse_train, label_train)
    
    fuse_result_train = dt.predict(fuse_train)
    print(fuse_result_train)
    fuse_result_dev = dt.predict(fuse_dev)
    fuse_result_test = dt.predict(fuse_test)

    devAccuracy = calAccuracy(dev_raw, fuse_result_dev)
    trainAccuracy = calAccuracy(train_raw, fuse_result_train)
    print("train acc =", trainAccuracy)
    print("dev acc =", devAccuracy)

    print("work completed")
    dump(fuse_result_test)

if __name__ == '__main__':
    main()
