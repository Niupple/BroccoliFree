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
qq2label = {}
qqReg = re.compile(r"[qQ][qQ]\:\d+")

cnt_true = 0
cnt_non = 0
cnt_find = 0
for text, label in data_dev:
    qq = qqReg.search(text)
    if qq is not None:
        if label == "1":
            cnt_true += 1
        else:
            print(text)
        cnt_find += 1
print(cnt_true, cnt_find, cnt_true / cnt_find)
