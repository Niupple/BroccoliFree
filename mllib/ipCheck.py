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

with open('../data/raw_train.csv', 'r', encoding='utf-8') as f:
    data_train = list(csv.reader(f))[1:]
with open('../data/dev.csv', 'r', encoding='utf-8') as f:
    data_dev = list(csv.reader(f))[1:]
ip2label = {}
ipReg = re.compile(r"\(\[\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\]\)")
for text, label in data_train:
    ip = ipReg.search(text)
    if ip is not None:
        ip2label[ip.group(0)] = label
cnt = 0
cnt_non = 0
cnt_find = 0
for text, label in data_dev:
    ip = ipReg.search(text)
    ans = -1
    if ip is not None:
        if ip.group(0) in ip2label:
            ans = ip2label[ip.group(0)]
            cnt_find += 1
    if ans == label:
        cnt += 1
print(cnt, cnt_find, len(data_dev), cnt / cnt_find)
