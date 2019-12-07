#encoding=utf-8
'''
Created on 2016年4月18日

@author: lenovo
'''
import jieba
import os
from math import log, fabs

jieba.initialize()

class spamEmailBayes:
    #获得停用词表
    def getStopWords(self):
        stopList=[]
        for line in open("中文停用词表.txt", encoding='gbk'):
            stopList.append(line[:len(line)-1])
        return stopList
    #获得词典
    def get_word_list(self,content,wordsList,stopList):
        #分词结果放入res_list
        res_list = list(jieba.cut(content))
        for i in res_list:
            if i not in stopList and i.strip()!='' and i!=None:
                wordsList.append(i)
                    
    #若列表中的词已在词典中，则加1，否则添加进去
    def addToDict(self,wordsList,wordsDict):
        for item in wordsList:
            wordsDict[item] = wordsDict.get(item, 0) + 1

    #通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
    def getTestWords(self,testDict,spamDict,normDict,normFilelen,spamFilelen):
        # print(spamDict, normDict)
        # print(normFilelen, spamFilelen)
        wordProbList={}
        default = 1/(1e30)
        # default = 0.00001
        for word,num  in testDict.items():
            #该文件中包含词个数
            pw_s=spamDict.get(word, default)/spamFilelen
            pw_n=normDict.get(word, default)/normFilelen
            ps_w=pw_s/pw_n
            # ps_w = log(ps_w)*num
            ps_w = log(ps_w)
            wordProbList.setdefault(word,ps_w)
        # wordProbList = sorted(wordProbList.items(),key=lambda d:d[1],reverse=True)[0:15]
        return (wordProbList)

    def calBayesLog(self,wordList,spamdict,normdict,normFilelen,spamFilelen):
        ret = log(spamFilelen/normFilelen)
        
        for word, prob_log in wordList.items() :
            ret += prob_log
        # print(str(ps_w)+"////"+str(ps_n))
        return ret
