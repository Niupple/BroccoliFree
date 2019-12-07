import csv
import random

TRAIN_PERCENT = 0.8
DEV_PERCENT = 1-TRAIN_PERCENT

def main():
    try:
        rinput = open('raw_train.csv', 'r')
        train_out = open('train.csv', 'w')
        dev_out = open('dev.csv', 'w')
        lines = list(csv.reader(rinput))[1:]
        random.shuffle(lines)
        n = len(lines)
        k = int(n*TRAIN_PERCENT)
        csv.writer(train_out).writerows(lines[:k])
        csv.writer(dev_out).writerows(lines[k:])
    except Exception as e:
        print(e)
        pass
    

if __name__=='__main__':
    main()