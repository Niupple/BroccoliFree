# encoding=utf8
import csv
import re
import jieba
import numpy as np
from multiprocessing import Pool
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.regression import LabeledPoint, array
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext
import io
import sys
from dataloader import load, label
from validator import valid, dump

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
    data_train_lp, data_dev_p, label_dev_gt, test_p = d['train_tfidf_lp'], d['dev_tfidf'], d['dev_gt'], d['test_tfidf']
    data_train_p, label_train_gt = d['train_tfidf'], d['train_gt']
    data_train, data_dev, data_test = d['train_raw'], d['dev_raw'], d['test_raw']

    data_train_lp = data_train_lp.sample(False, 0.01)
    
    # print(sum(data_train_lp.first()[0]))
    # print(data_train_lp.zipWithIndex().collect())
    print(data_train_lp.take(2))
    print("___________train_bayes_____________")
    sys.stdout.flush()
    nb = NaiveBayes.train(data_train_lp)
    print("___________trained_bayes___________")
    sys.stdout.flush()
    # nb.save(sc, 'bayes.model')
    bayes_result_dev = nb.predict(data_dev_p).map(int)
    bayes_result_dev.count()
    bayes_result_train = nb.predict(data_train_p).map(int)
    bayes_result_train.count()
    bayes_result_test = nb.predict(test_p).map(int)
    bayes_result_test.count()
    
    print("train info:")
    valid(bayes_result_train, label_train_gt)
    print("dev info:")
    valid(bayes_result_dev, label_dev_gt)

    print("___________train_logistic_____________")
    sys.stdout.flush()
    lg = LogisticRegressionWithSGD.train(data_train_lp, step=0.005)
    print("___________trained_logisitc___________")
    sys.stdout.flush()
    # lg.save(sc, 'logistic.model')
    logistic_result_dev = lg.predict(data_dev_p).map(int)
    logistic_result_train = lg.predict(data_train_p).map(int)
    logistic_result_test = lg.predict(test_p).map(int)

    print("train info:")
    valid(logistic_result_train, label_train_gt)
    print("dev info:")
    valid(logistic_result_dev, label_dev_gt)

    fused_train_p = stack_label([bayes_result_train, logistic_result_train])
    fused_dev_p = stack_label([bayes_result_dev, logistic_result_dev])
    fused_test_p = stack_label([bayes_result_test, logistic_result_test])

    fused_train_lp = label(data_train, fused_train_p)

    print("___________train_GBDT___________")
    sys.stdout.flush()
    gbdt = GradientBoostedTrees.trainClassifier(fused_train_lp, {})
    print('___________trained_GBDT_________')
    sys.stdout.flush()

    fused_result_train = gbdt.predict(fused_train_p)
    fused_result_dev = gbdt.predict(fused_dev_p)
    fused_result_test = gbdt.predict(fused_test_p)

    print("train info:")
    valid(fused_result_train, label_train_gt)
    print("dev info:")
    valid(fused_result_dev, label_dev_gt)

    dump(fused_result_test.map(int).collect())

    sc.show_profiles()


if __name__ == "__main__":
    print("特别优秀")
    main()
