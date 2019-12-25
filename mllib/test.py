import io
import sys
import jieba
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

jieba.initialize()

s = '这是一段没有意义的但是合乎语法的中文句子用来测试结巴工具的分词功能'
print(list(jieba.cut(s)))