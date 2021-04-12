import csv
import re
import jieba
import numpy as np
from multiprocessing import Pool
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.tree import GradientBoostedTreesModel, RandomForestModel, GradientBoostedTrees
from pyspark.mllib.regression import LabeledPoint, array
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext
import io
import sys
from dataloader import load, label
from validator import valid, dump
from myBayes import MyBayes
from ipCheckRdd import ipJudge
from qqCheckRdd import qqCheck

def main():
    sc = SparkContext('local[15]', 'haha')
    d = load(sc)

    data_train, data_dev, data_test = d['train_raw'], d['dev_raw'], d['test_raw']
    label_train_gt, label_dev_gt = d['train_gt'], d['dev_gt']

    gbdt = GradientBoostedTreesModel.load(sc, 'gbdt.model')

if __name__ == "__main__":
    main()