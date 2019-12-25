# encoding=utf8
import csv
import re
import jieba
import numpy as np
import math
from multiprocessing import Pool
from pyspark.mllib.regression import LabeledPoint, array
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

jieba.initialize()

def textwithlabel2words(data : (str, str)):
    text, label = data
    rule = re.compile(r"[^\u4e00-\u9fa5]")
    line = rule.sub('', text)
    line = list(jieba.cut(line))
    # print(list(line))
    return (line, int(label))
    # return linel

def text2words(data : (str, )):
    text = data[0]
    rule = re.compile(r"[^\u4e00-\u9fa5]")
    line = rule.sub('', text)
    line = list(jieba.cut(line))
    return (line, )

def words2vec(words : list, dictionary):
    ret = {}
    for word in words:
        if word in dictionary:
            ret[dictionary[word]] = ret.get(dictionary[word], 0) + 1
    lst = sorted(ret.items(), key=lambda x : (x[0]))
    return Vectors.sparse(len(dictionary), [i[0] for i in lst], [i[1] for i in lst])

def read_from_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.reader(f))[1:]

def label(pair, point):
    return pair.zip(point).map(lambda x : LabeledPoint(x[0][1], x[1]))

def num2rate(sv):
    s = 0
    for x in sv.indices:
        s += sv[x]
    return Vectors.sparse(sv.numNonzeros(), sv.indices, [sv[i]/s for i in sv.indices])

def num2tf(num, mode='cnt'):
    if mode == 'cnt':
        return num
    elif mode == 'rate':
        return num/max(1, num.sum())
    elif mode == 'bool':
        return array([0. if int(x) == 0 else 1. for x in num])
    else:
        raise NotImplementedError()

def load(sc):
    data_train = read_from_file('../data/train.csv')
    data_dev = read_from_file('../data/dev.csv')
    data_test = read_from_file('../data/test.csv')

    data_train = sc.parallelize(data_train).sample(False, 0.01)
    data_dev = sc.parallelize(data_dev)
    data_test = sc.parallelize(data_test)

    print('train_cnt = ', data_train.count())
    # print(data_train.map(lambda x : [x[0][:min(len(x[0]), 100)], x[1]]).zipWithIndex().collect())

    data_train_count = data_train.count()

    data_train = data_train.map(textwithlabel2words)
    data_dev = data_dev.map(textwithlabel2words)
    data_test = data_test.map(text2words)

    # print(data_train.take(10))

    d : dict = data_train.flatMap(lambda x : x[0]).zipWithIndex() \
        .countByKey()
    # print(d)
    d = list(map(lambda x : x[0], filter(lambda x : x[1] > 2, d.items())))
    global dictionary
    dictionary = {word : idx for idx, word in enumerate(sorted(d))}
    print(dictionary)

    # from document (List(word)) to Point (SparseVector)
    doc2point = lambda x : array(words2vec(x[0], dictionary))

    data_train_p = data_train.map(doc2point)
    data_dev_p = data_dev.map(doc2point)
    data_test_p = data_test.map(doc2point)

    num2rate = lambda x : x

    # from frequency in number to frequency in ratio
    data_train_tf = data_train_p.map(num2rate)
    data_dev_tf = data_dev_p.map(num2rate)
    data_test_tf = data_test_p.map(num2rate)

    # calculate term appearance in docs
    d = data_train \
        .flatMap(lambda x : set(x[0])).zipWithIndex() \
        .countByKey()
    idf = array([math.log(data_train_count/d[x]) for x in dictionary.keys()])

    print('idf caculated')
    sys.stdout.flush()

    get_tfidf = lambda x : x * idf

    data_train_tfidf = data_train_tf.map(get_tfidf)
    data_dev_tfidf = data_dev_tf.map(get_tfidf)
    data_test_tfidf = data_test_tf.map(get_tfidf)

    data_train_lp = label(data_train, data_train_p)
    data_dev_lp = label(data_dev, data_dev_p)
    label_dev_gt = data_dev.map(lambda x : int(x[1]))
    label_train_gt = data_train.map(lambda x : int(x[1]))

    data_train_tfidf_lp = label(data_train, data_train_tfidf)
    data_dev_tfidf_lp = label(data_dev, data_dev_tfidf)

    # print(data_train_tfidf_lp.zip(data_train).collect())

    ret = {}
    ret['train_lp'] = data_train_lp
    ret['train_p'] = data_train_p
    ret['train_gt'] = label_train_gt
    ret['train_tfidf'] = data_train_tfidf
    ret['train_tfidf_lp'] = data_train_tfidf_lp
    ret['dev_tfidf'] = data_dev_tfidf
    ret['dev_tfidf_lp'] = data_dev_tfidf_lp
    ret['dev_lp'] = data_dev_lp
    ret['dev_p'] = data_dev_p
    ret['dev_gt'] = label_dev_gt
    ret['test_tfidf'] = data_test_tfidf
    ret['test_p'] = data_test_p

    print('data loaded!')
    sys.stdout.flush()

    return ret

