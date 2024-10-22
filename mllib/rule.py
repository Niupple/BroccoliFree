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
#-1 normal 1 rubbish
class rule:
    def __init__(self):
        self.d = {}
        self.qqReg = re.compile(r"[qQ][qQ]\s*\:\s*\d+")
        self.ipReg = re.compile(r"\(\[\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\]\)")

    def qqPredict(self, text : str):
        qq = self.qqReg.search(text)
        if qq is not None:
            return 1
        return 0
    
    def ipTrain(self, data_train):
        for text, label in data_train:
            ip = self.ipReg.search(text)
            if ip is not None:
                self.d[ip.group(0)] = label
    
    def ipPredict(self, data_predict):
        ans = []
        for text in data_predict:
            ip = self.ipReg.search(text)
            label = -1
            if ip is not None:
                if ip.group(0) in self.d:
                    label = int(self.d[ip.group(0)])
            ans.append(label)
        return ans

r = rule()
