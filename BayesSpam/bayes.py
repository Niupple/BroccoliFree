# encoding=utf-8
'''
Created on 2016年4月18日

@author: lenovo
'''

import re
import csv
import pickle
from multiprocessing import Pool
from collections import Counter
from functools import partial
from math import fabs, log
import jieba

jieba.initialize()



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
        if int(devs[i][1]) == outs[i][0]:
            equ += 1
        elif devs[i][1] == '0':
            fake_true += 1
            print(devs[i][0], devs[i][1], outs[i][0], outs[i][1], file=ftrue)
        else:
            fake_false += 1
            print(devs[i][0], devs[i][1], outs[i][0], outs[i][1], file=ffalse)
        if fabs(outs[i][1]) < 0.2:
            print(devs[i][0], devs[i][1], outs[i][0], outs[i][1], file=fclose)
    print("fake true = %f(%d), fake false = %f(%d)" %
          (fake_true/n, fake_true, fake_false/n, fake_false))
    return equ/n


def joinmaps(lst):
    ret = {}
    cnt = 0
    for d in lst:
        for k, v in d.items():
            ret[k] = ret.get(k, 0) + v
            cnt += v
    return ret, cnt

class BayesSpam:
    def __init__(self):
        self.norm_file_len = 0
        self.spam_file_len = 0
        self.norm_dict = {}
        self.spam_dict = {}
        self.words_dict = {}
        self.stop_list = []
        self.words_list = []

        with open("stoplist.txt", encoding='utf-8') as sl:
            for line in sl:
                self.stop_list.append(line)

    def train(self, data):
        spams = []
        normals = []
        for line in data:
            if line[1] == '1':
                spams.append(line[0])
            else:
                normals.append(line[0])
        self.norm_file_len = len(normals)
        self.spam_file_len = len(spams)
        print("normal email number: ", self.norm_file_len)
        print("spam email number: ", self.spam_file_len)
        self.norm_dict = self._update_words_to_dict(normals)
        self.spam_dict = self._update_words_to_dict(spams)

    def predict(self, data):
        pool = Pool()
        return pool.map(self._test_line, data)

    def _update_words_to_dict(self, emails):
        pool = Pool()
        dicts = pool.map(self._load_email, emails)
        return joinmaps(dicts)[0]

    def _test_line(self, email):
        text = email[0]
        rule = re.compile(r"[^\u4e00-\u9fa5]")
        text = rule.sub("", text)
        wl = self._load_email(text)
        p = self._dict_to_prob(wl)
        if(p > 0):
            return 1, p
        else:
            return 0, p

    def _dict_to_prob(self, testDict):
        ret = log(self.spam_file_len/self.norm_file_len)
        # print(spamDict, normDict)
        # print(normFilelen, spamFilelen)
        default = 1/(1e30)
        # default = 0.00001
        for word in testDict:
            pw_s = self.spam_dict.get(word, default)/self.spam_file_len
            pw_n = self.norm_dict.get(word, default)/self.norm_file_len
            ps_w = pw_s/pw_n
            # ps_w = log(ps_w)*num
            ps_w = log(ps_w)
            ret += ps_w
        # wordProbList = sorted(wordProbList.items(),key=lambda d:d[1],reverse=True)[0:15]
        return ret

    def _load_email(self, normal):
        word_list = []
        word_dict = {}
        line = normal
        rule = re.compile(r"[^\u4e00-\u9fa5]")
        line = rule.sub("", line)
        res_list = list(jieba.cut(line))
        for i in res_list:
            if i not in self.stop_list and i.strip() != '' and i != None:
                word_dict[i] = word_dict.get(i, 0) + 1

        return word_dict

    def _add_to_dict(self, wordsList, wordsDict):
        for item in wordsList:
            wordsDict[item] = wordsDict.get(item, 0) + 1

def dump(rets):
    with open('answer.txt', 'w') as f:
        print("\n".join(map(str, map(lambda x: x[0], rets))), file=f)

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

    result_train = bs.predict(train_raw)
    result_dev = bs.predict(dev_raw)
    result_test = bs.predict(test_raw)
    print("test completed")

    devAccuracy = calAccuracy(dev_raw, result_dev)
    trainAccuracy = calAccuracy(train_raw, result_train)
    print("train acc =", trainAccuracy)
    print("dev acc =", devAccuracy)
    print("_______bayes________")

    print("work completed")
    dump(result_test)

if __name__ == '__main__':
    main()
