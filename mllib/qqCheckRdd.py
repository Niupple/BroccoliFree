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
qqReg = re.compile(r"[qQ][qQ]\s*\:\s*\d+")
#-1 normal 1 rubbish
def qqCheck(data : (str, )):
    qq = qqReg.search(data[0])
    if qq is not None:
        return 1
    return 0

def read_from_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.reader(f))[1:]

if __name__ == "__main__":
    sc = SparkContext('local[3]', 'haha')
    data_dev = read_from_file('../data/dev.csv')
    data_dev = sc.parallelize(data_dev)
    data_dev = data_dev.map(qqCheck)
    ans = data_dev.collect()
    print(ans)