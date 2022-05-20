'''
1. 利用给定语料库(或者自选语料库),利用神经语言模型(如:Word2Vec, GloVe等模型)来训练词向量.
2. 通过对词向量的聚类或者其他方法来验证词向量的有效性.
'''

import jieba
import os
import gensim
from gensim import models
import re

class Doc:
    lable = ''
    wordList = []
    topicDistribute = []
    sentences = []

    def __init__(self, name, wordList, sentences):
        self.lable = name
        self.wordList = wordList
        self.sentences = sentences

    def setTopicDist(self, topicDist):
        self.topicDistribute = topicDist

def readNovels(path):
    # 16篇小说，即16个文章
    articles = []
    names = os.listdir(path)
    for name in names:
        novelPath = path + '\\' + name
        with open(novelPath, 'r', encoding='UTF-8') as f:
            text = f.read()
            sentencesString =re.split("[。？！.?!]", filter(text))
            words = list(jieba.lcut(filter(text)))
            sentences = []
            for sentence in sentencesString:
                sentences.append(list(jieba.lcut(sentence)))
            articles.append(Doc(name.replace('.txt', ''), words, sentences))
        f.close()
    return articles

def filter(novel):
    strs = ['，', '；', '：', '、', '《', '》', '“', '”',
          '‘', '’', '［', '］', '....', '......', '『', '』', '（', '）', '…', '「', '」',
          '＜', '＞', '+', '【', '】', '(', 'com', 'cr173', ')', 'www', '=']
    for str in strs:
        novel = novel.replace(str, '')
    return novel

def takeSecond(elem):
    return elem[1]

def generateModel():
    articles = readNovels('./database')
    wordsDocuments = [article.wordList for article in articles]
    allWordsList = []
    for words in wordsDocuments:
        allWordsList += words
    allSentences = []
    sentenceDocuments = [article.sentences for article in articles]
    for i in range(len(sentenceDocuments)):
        allSentences += sentenceDocuments[i]
    wordsFreq = {}
    for word in allWordsList:
        if wordsFreq.get(word) == None:
            wordsFreq[word] = 1
        else:
            wordsFreq[word] = wordsFreq[word] + 1
    wordsFreqList = list(wordsFreq.items())
    wordsFreqList.sort(key=takeSecond, reverse=True)
    model = gensim.models.Word2Vec(sentences=allSentences, vector_size=100, window=5, sg=1)
    model.save('./word2vec2.model')

if __name__ == '__main__':
    generateModel()
    model = models.Word2Vec.load('word2vec2.model')
    print(model.wv['令狐冲'])
    print(model.wv.most_similar('令狐冲'))
    print(model.wv.most_similar('屠龙刀'))
    print(model.wv.most_similar('明教'))
    print(model.wv.most_similar('华山'))
    print(model.wv.most_similar('逃走'))