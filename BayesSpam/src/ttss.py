#encoding=utf-8
'''
Created on 2016年4月18日

@author: lenovo
'''

from spam.spamEmail import spamEmailBayes
import re
import csv
import pickle
from multiprocessing import Pool
from collections import Counter
from functools import partial
spam=spamEmailBayes()
#保存词频的词典
spamDict={}
normDict={}
#保存每封邮件中出现的词
wordsDict={}
#保存预测结果,key为文件名，值为预测类别
#分别获得正常邮件、垃圾邮件及测试文件名称列表
#获取训练集中正常邮件与垃圾邮件的数量
normFilelen=0
spamFilelen=0
#获得停用词表，用于对停用词过滤
stopList=spam.getStopWords()
#获得正常邮件中的词频

all_in_one = []
normals = []
spams = []
devs = []
outs = []


def load_normal(normal):
    wordsDict = {}
    wordsList = []
    for line in normal.split('\n'):
        #过滤掉非中文字符
        rule=re.compile(r"[^\u4e00-\u9fa5]")
        line=rule.sub("",line)
        #将每封邮件出现的词保存在wordsList中
        spam.get_word_list(line,wordsList,stopList)
    #统计每个词在所有邮件中出现的次数
    spam.addToDict(wordsList, wordsDict)
    return wordsDict

def load_spam(spami):
    wordsDict = {}
    wordsList = []
    for line in spami.split('\n'):
        rule=re.compile(r"[^\u4e00-\u9fa5]")
        line=rule.sub("",line)
        spam.get_word_list(line,wordsList,stopList)
    spam.addToDict(wordsList, wordsDict)
    return wordsDict

def calc_test(test, spamDict, normDict, normFilelen, spamFilelen):
    testDict = {}
    wordsDict = {}
    wordsList = []
    for line in test[0].split('\n'):
        rule=re.compile(r"[^\u4e00-\u9fa5]")
        line=rule.sub("",line)
        spam.get_word_list(line,wordsList,stopList)
    spam.addToDict(wordsList, wordsDict)
    testDict=wordsDict.copy()
    #通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
    wordProbList=spam.getTestWords(testDict, spamDict,normDict,normFilelen,spamFilelen)
    #对每封邮件得到的15个词计算贝叶斯概率  
    p=spam.calBayesLog(wordProbList, spamDict, normDict, normFilelen,spamFilelen)
    # if p > 0.1:
        # print(p)
    if(p>0):
        # outs.append(1)
        return 1
    else:
        # outs.append(0)
        return 0

def calAccuracy(devs, outs):
    n = len(outs)
    equ = 0
    fake_true = 0
    fake_false = 0
    # assert len(devs) == len(outs)
    for i in range(n):
        if int(devs[i][1]) == outs[i]:
            equ += 1
        elif devs[i][1] == '0':
            fake_true += 1
            print(devs[i][0], devs[i][1], outs[i])
        else:
            fake_false += 1
    print("fake true = %f, fake false = %f" % (fake_true/n, fake_false/n))
    return equ/n

def joinmaps(lst):
    ret = {}
    for d in lst:
        for k, v in d.items():
            ret[k] = ret.get(k, 0) + v
    return ret

def main():
    #spam类对象

    with open("../../Chinese_spam_mails/train.csv", 'r', encoding='utf-8') as f:
        all_in_one = list(csv.reader(f))
        for line in all_in_one:
            if line[1] == '1':
                spams.append(line[0])
            else:
                normals.append(line[0])
        global normFilelen, spamFilelen
        normFilelen = len(normals)
        spamFilelen = len(spams)
        print("loaded %d normal emails and %d spams" % (len(normals), len(spams)))

    with open("../../Chinese_spam_mails/dev.csv", 'r', encoding='utf-8') as f:
        devs = list(csv.reader(f))

    tests = []

    with open("../../Chinese_spam_mails/test.csv", 'r', encoding='utf-8') as f:
        tests = list(next(csv.reader(f)))

    pool = Pool()

    global normDict, spamDict, wordsDict

    wordsDict.clear()

    nds = pool.map(load_normal, normals)
    normDict=joinmaps(nds)
    print("normal email loaded")

    #获得垃圾邮件中的词频
    wordsDict.clear()

    sds = pool.map(load_spam, spams)
    spamDict=joinmaps(sds)
    print("spam email loaded")

    print(sorted(normDict.items(), key=lambda x : (-x[1]))[:20])
    print(sorted(spamDict.items(), key=lambda x : (-x[1]))[:20])

    outs = pool.map(partial(calc_test, spamDict=spamDict, normDict=normDict, normFilelen=normFilelen, spamFilelen=spamFilelen), devs)
    print("test completed") 

    testAccuracy=calAccuracy(devs, outs)
    # for i,ic in testResult.items():
    #     print(i+"/"+str(ic))
    print(testAccuracy)
    
    rets = pool.map(partial(calc_test, spamDict=spamDict, normDict=normDict, normFilelen=normFilelen, spamFilelen=spamFilelen), tests)
    print("work completed")
    with open('answer.txt', 'w') as f:
        print("\n".join(map(str, rets)), file=f)

if __name__=='__main__':
    main()