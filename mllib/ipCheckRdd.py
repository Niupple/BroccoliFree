import csv
import re
import jieba
import numpy as np
from multiprocessing import Pool
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint, array
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext
import io
import sys
from dataloader import load
from validator import valid, dump
from functools import partial

ipReg = re.compile(r"\(\[\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\]\)")
#def ipCheckRdd(d : dict):
#    def _ipCheckRdd(data : (str, )):
#        ip = ipReg.search(data[0])
#    return _ipCheckRdd

def ipCheckRddPartial(data : (str, ), d : dict):
    ip = ipReg.search(data[0])
    if ip is not None:
        if ip.group(0) in d:
            return 1 if d[ip.group(0)] == "0" else -1
    return 0
    
def ipCheckRdd(data : ((str, str), int)):
    ip = ipReg.search(data[0][0])
    if ip is not None:
        return (data[1], (ip.group(0), data[0][1]))
    return (data[1], ("0.0.0.0", data[0][1]))
        
def seqOp(a, b):
    a[b[0]] = b[1]
    return a

#-1 normal 1 rubbish
class ipJudge:
    def __init__(self):
        self.d = {}
    def train(self, rdd):
        rdd = rdd.zipWithIndex().map(ipCheckRdd).aggregateByKey(dict(), seqOp, lambda x, y : x.update(y))
        ans = rdd.collect()
        for x, y in ans:
            self.d.update(y)
    def predict(self, rdd):
        rdd = rdd.map(partial(ipCheckRddPartial, d=self.d))
        return rdd

def read_from_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.reader(f))[1:]

if __name__ == "__main__":
    sc = SparkContext('local[3]', 'haha')
    data_train = read_from_file('../data/train.csv')
    data_dev = read_from_file('../data/dev.csv')
    data_train = sc.parallelize(data_train)
    data_dev = sc.parallelize(data_dev)
    ipJ = ipJudge()
    ipJ.train(data_train)

    data_dev = ipJ.predict(data_dev)
    ans = data_dev.collect()
    print(ans)