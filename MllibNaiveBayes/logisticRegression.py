from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark import SparkContext
from dataloader import load
from validator import valid, dump
import sys

def main():
    sc = SparkContext('local[15]', 'haha')
    d = load(sc)
    data_train_lp, data_dev_p, label_dev_gt, test_p = d['train_lp'], d['dev_p'], d['dev_gt'], d['test_p']
    data_train_p, label_train_gt = d['train_p'], d['train_gt']
    print("count =", data_train_lp.take(10))
    sample_train = data_train_lp
    print("sample in total: ", sample_train.count())
    print("___________train______________")
    sys.stdout.flush()
    lg = LogisticRegressionWithSGD.train(sample_train, step=0.000001)
    print("___________trained____________")
    sys.stdout.flush()
    lg.save(sc, 'logistic.model')
    result_dev = lg.predict(data_dev_p).map(int)
    result_train = lg.predict(data_train_p).map(int)

    print("train info:")
    valid(result_train, label_train_gt)
    print("dev info:")
    valid(result_dev, label_dev_gt)
    dump(lg.predict(test_p).map(int).collect())

if __name__ == "__main__":
    main()
