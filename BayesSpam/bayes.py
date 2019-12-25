import re
from math import log
from multiprocessing import Pool

import jieba

jieba.initialize()

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
            return 1
        else:
            return 0

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
