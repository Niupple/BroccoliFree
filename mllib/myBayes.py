from pyspark.rdd import RDD
import jieba
from math import log
import re

class MyBayes:
    def __init__(self):
        self.norm_file_len = 0
        self.spam_file_len = 0
        self.norm_dict = {}
        self.spam_dict = {}
        self.words_dict = {}
        self.stop_list = set()
        self.words_list = []

        with open("stoplist.txt", encoding='utf-8') as sl:
            for line in sl:
                self.stop_list.add(line)

    def train(self, data : RDD):
        spams = data.filter(lambda x : x[1] == '1').map(lambda x : x[0])
        normals = data.filter(lambda x : x[1] == '0').map(lambda x : x[0])
        self.norm_file_len = normals.count()
        self.spam_file_len = spams.count()
        print("normal email number: ", self.norm_file_len)
        print("spam email number: ", self.spam_file_len)
        self.norm_dict = self._cut_test_to_dict(normals)
        self.spam_dict = self._cut_test_to_dict(spams)

    def predict(self, data : RDD):
        return data.map(self._test_line)

    def _test_line(self, email : (str, )):
        word_dict = {}
        text = email[0]
        rule = re.compile(r"[^\u4e00-\u9fa5]")
        text = rule.sub("", text)
        rlist = list(jieba.cut(text))
        for i in rlist:
            if i not in self.stop_list and i.strip() != '' and i is not None:
                word_dict[i] = word_dict.get(i, 0) + 1
        ret = log(self.spam_file_len/self.norm_file_len)
        default = 1/(1e30)
        for word in word_dict:
            pw_s = self.spam_dict.get(word, default)/self.spam_file_len
            pw_n = self.norm_dict.get(word, default)/self.norm_file_len
            ps_w = pw_s/pw_n
            # ps_w = log(ps_w)*num
            ps_w = log(ps_w)
            ret += ps_w
        # wordProbList = sorted(wordProbList.items(),key=lambda d:d[1],reverse=True)[0:15]
        if ret > 0:
            return 1
        else:
            return 0

    def _cut_test_to_dict(self, emails : RDD):
        dicts = emails \
            .map(self._wash_text) \
            .flatMap(lambda x : list(jieba.cut(x))) \
            .filter(lambda x : x not in self.stop_list) \
            .zipWithIndex() \
            .countByKey()
        return dicts

    def _wash_text(self, text : str):
        rule = re.compile(r"[^\u4e00-\u9fa5]")
        line = rule.sub("", text)
        return line

