# encoding=utf8
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

def main():
    sc = SparkContext('local[15]', 'haha')

    d = load(sc)
    data_train_lp, data_dev_p, label_dev_gt, test_p = d['train_tfidf_lp'], d['dev_tfidf'], d['dev_gt'], d['test_tfidf']
    data_train_p, label_train_gt = d['train_tfidf'], d['train_gt']
    
    # print(sum(data_train_lp.first()[0]))
    # print(data_train_lp.zipWithIndex().collect())
    print(data_train_lp.take(2))
    print("___________train______________")
    sys.stdout.flush()
    nb = NaiveBayes.train(data_train_lp)
    print("___________trained____________")
    sys.stdout.flush()
    # nb.save(sc, 'bayes.model')
    result_dev = nb.predict(data_dev_p).map(int)
    result_dev.count()
    result_train = nb.predict(data_train_p).map(int)
    result_train.count()
    result_test = nb.predict(test_p).map(int)
    result_test.count()
    
    print("train info:")
    valid(result_train, label_train_gt)
    print("dev info:")
    valid(result_dev, label_dev_gt)
    dump(result_test.collect())


if __name__ == "__main__":
    print("特别优秀")
    main()
