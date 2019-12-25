import sys
from operator import add

def valid(result_dev, label_dev_gt):
    n = result_dev.count()
    assert result_dev.count() == label_dev_gt.count()
    eq = result_dev.zip(label_dev_gt).map(lambda x : 1 if int(x[0]) == int(x[1]) else 0)
    cnt = eq.reduce(add)
    print("accuracy:", cnt/n)
    sys.stdout.flush()

def dump(test_label):
    with open('answer.txt', 'w') as f:
        print('\n'.join(map(str, test_label)), file=f)
