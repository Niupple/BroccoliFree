# encoding=utf8
import csv
import re
import jieba
import numpy as np
from multiprocessing import Pool
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.tree import GradientBoostedTrees, RandomForest, GradientBoostedTreesModel
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

def stack_label(lst):
    assert len(lst) > 0
    ret = lst[0].map(lambda x : [x, ])
    for i in range(1, len(lst)):
        ret = ret.zip(lst[i]).map(lambda x : [*x[0], x[1]])
    return ret

def main():
    sc = SparkContext('local[15]', 'haha')
    # sc._conf.set("spark.python.profile", "true")

    print(sc.getConf().getAll())

    d = load(sc)
    data_train, data_dev, data_test = d['train_raw'], d['dev_raw'], d['test_raw']
    label_train_gt, label_dev_gt = d['train_gt'], d['dev_gt']


    nb = MyBayes()
    print(data_train.take(2))
    print("___________train_bayes_____________")
    sys.stdout.flush()
    nb.train(data_train)
    print("___________trained_bayes___________")
    sys.stdout.flush()
    # nb.save(sc, 'bayes.model')
    bayes_result_dev = nb.predict(data_dev).map(int)
    bayes_result_dev.count()
    bayes_result_train = nb.predict(data_train).map(int)
    bayes_result_train.count()
    bayes_result_test = nb.predict(data_test).map(int)
    bayes_result_test.count()

    print(label_train_gt.take(10))
    print(bayes_result_train.take(10))
    
    print("train info:")
    valid(bayes_result_train, label_train_gt)
    print("dev info:")
    valid(bayes_result_dev, label_dev_gt)

    ip = ipJudge()
    print("___________train_ip_____________")
    sys.stdout.flush()
    ip.train(data_train)
    print("___________trained_ip___________")

    ip_result_dev = ip.predict(data_dev).map(int)
    ip_result_dev.count()
    ip_result_train = ip.predict(data_train).map(int)
    ip_result_train.count()
    ip_result_test = ip.predict(data_test).map(int)
    ip_result_test.count()

    qq_result_dev = data_dev.map(qqCheck).map(int)
    qq_result_dev.count()
    qq_result_train = data_train.map(qqCheck).map(int)
    qq_result_train.count()
    qq_result_test = data_test.map(qqCheck).map(int)
    qq_result_test.count()

    fused_train_p = stack_label([bayes_result_train, ip_result_train, qq_result_train])
    fused_dev_p = stack_label([bayes_result_dev, ip_result_dev, qq_result_dev])
    fused_test_p = stack_label([bayes_result_test, ip_result_test, qq_result_test])

    fused_train_lp = label(data_train, fused_train_p)

    print("___________train_GBDT___________")
    sys.stdout.flush()
    # gbdt = GradientBoostedTrees.trainClassifier(fused_train_lp, {})
    gbdt = GradientBoostedTreesModel.load(sc, 'gbdt.model')
    # gbdt = RandomForest.trainClassifier(fused_train_lp, 2, )
    print('___________trained_GBDT_________')
    sys.stdout.flush()

    # gbdt.save(sc, 'gbdt.model')

    # fused_result_train = gbdt.predict(fused_train_p)
    fused_result_dev = gbdt.predict(fused_dev_p)
    fused_result_test = gbdt.predict(fused_test_p)

    # print("train info:")
    # valid(fused_result_train, label_train_gt)
    print("dev info:")
    # valid(fused_result_dev, label_dev_gt)

    dump(fused_result_test.map(int).collect())
    print("dumped")
    sc.show_profiles()


if __name__ == "__main__":
    print("特别优秀")
    main()
