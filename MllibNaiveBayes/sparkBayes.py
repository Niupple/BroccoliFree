# encoding=utf8
import csv
import re
import jieba
import numpy as np
from multiprocessing import Pool
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.regression import LabeledPoint, array
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

jieba.initialize()

dictionary = {}

def text2words(data : (str, str)):
    text, label = data
    rule = re.compile(r"[^\u4e00-\u9fa5]")
    line = rule.sub('', text)
    line = list(jieba.cut(line))
    # print(list(line))
    return (line, label)
    # return linel

def words2vec(words : list):
    ret = [0 for i in range(len(dictionary))]
    for word in words:
        if word in dictionary:
            ret[dictionary[word]] += 1
    return ret

def main():
    with open('../data/raw_train.csv', 'r', encoding='utf-8') as f:
        print('______open_completed_____')
        data_train = list(csv.reader(f))[1:]
    with open('../data/dev.csv', 'r', encoding='utf-8') as f:
        data_dev = list(csv.reader(f))[1:]
    sc = SparkContext('local[*]', 'haha')
    data_train = sc.parallelize(data_train)
    data_dev = sc.parallelize(data_dev)

    data_train = data_train.map(text2words)
    data_dev = data_dev.map(text2words)

    d : list = data_train.flatMap(lambda x : x[0]) \
        .distinct() \
        .collect()
    global dictionary
    dictionary = {word : idx for idx, word in enumerate(sorted(d))}
    # print(dictionary)

    data_train_lp = data_train.map(lambda x : LabeledPoint(x[1], words2vec(x[0])))
    # data_dev_lp = data_dev.map(lambda x : LabeledPoint(x[1], words2vec(x[0])))
    label_dev_gt = data_dev.map(lambda x : int(x[1])).collect()

    data_dev_p = data_dev.map(lambda x : array(words2vec(x[0])))

    # print(sum(data_train_lp.first()[0]))
    print("___________train______________")
    sys.stdout.flush()
    nb = NaiveBayes.train(data_train_lp)
    print("___________train______________")
    result_dev = nb.predict(data_dev_p).map(int).collect()

    n = len(result_dev)
    cnt = 0
    assert len(result_dev) == len(label_dev_gt)
    print(result_dev)
    print(label_dev_gt)
    for x, y in zip(result_dev, label_dev_gt):
        if x == y:
            cnt += 1
    print("accuracy:", cnt/n)

if __name__ == "__main__":
    print("特别优秀")
    main()
