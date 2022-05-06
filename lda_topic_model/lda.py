from io import TextIOWrapper
from time import time

'''
LDA (Latent Dirichlet Allocation)
1. 从给定的语料库中均匀抽取200个段落(每个段落大于500个词)
2. 每个段落的标签就是对应段落所属的小说
3. 利用LDA(Latent Dirichlet Allocation)主题模型进行文本建模,并把每个段落表示为主题分布后进行分类.
'''

import os
from typing import List, Tuple
import jieba
import gensim
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from gensim.models.ldamodel import LdaModel

class FileHandlerIterator:
  READ_BATCH_MAX = 100
  TERMINATOR_CHAR_LIST = ['.', '。', '?', '？', ';', '；', '...']

  def __init__(self, f: TextIOWrapper):
    self.buf = ''
    self.f = f
    return

  def __iter__(self):
    return self

  def __next__(self):
    if self.f.closed:
      raise StopIteration
    else: 
      pos, char = self.search()
      if pos >= 0 and char:
        line = self.buf[: pos]
        self.buf = self.buf[pos + len(char):]
        return line
      else:
        chunk = self.f.read(self.READ_BATCH_MAX)
        if chunk:
          self.buf += chunk
          return self.__next__()
        else:
          self.f.close()
          raise StopIteration

  def search(self):
    dict = {}
    for char in self.TERMINATOR_CHAR_LIST:
      try:
        pos = self.buf.index(char)
        dict[char] = pos
      except:
        dict[char] = -1
    res = (-1, None)
    for char in dict:
      pos = dict[char]
      if pos > -1:
        res = (pos, char)
    return res

class FileHandler:
  def __init__(self, filePath: str) -> None:
    self.buf = ''
    fileName = os.path.basename(filePath).split('.')[0]
    self.fileName = fileName
    f = open(filePath, encoding='utf-8')
    self.f = f
    return

  def readLine(self):
    if not self.f.closed:
      return FileHandlerIterator(self.f)
    else:
      raise Exception('文件已关闭. 某一文本已遍历完毕, 取词数超出范围。')

  def close(self):
    if not self.f.closed:
      self.f.close()
    return

  

class FileListHanlder:
  def __init__(self) -> None:
    pass

  def getFilePathFromCorpus(self) -> List[str]:
    list = []
    dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
    for filename in os.listdir(dirname):
      list.append(os.path.join(dirname, filename))
    return list
  
  def getFileHandlers(self, filePathList: List[str]) -> List[FileHandler]:
    list = []
    for filePath in filePathList:
      fileHandler = FileHandler(filePath)
      list.append(fileHandler)
    return list
    
  def sampleFileList(self, n: int, length: int, FileHandlers: List[FileHandler]) -> List[Tuple[str, List[str]]]:
    count = 0
    list = []
    while(count < n):
      for fileHandler in FileHandlers:
        if count >= n:
          break
        wordList = []
        fileName= fileHandler.fileName
        for line in fileHandler.readLine():
          if len(wordList) > length:
            break
          for word in jieba.cut(line, cut_all=False):
            if self.isChinese(word):
              wordList.append(word)
        list.append((fileName, wordList))
        count += 1
    return list

  def getAllFile(self, FileHandlers: List[FileHandler]) -> List[Tuple[str, List[str]]]:
    list = []
    for fileHandler in FileHandlers:
      wordList = []
      fileName= fileHandler.fileName
      for line in fileHandler.readLine():
        for word in jieba.cut(line, cut_all=False):
          if self.isChinese(word):
            wordList.append(word)
      list.append((fileName, wordList))
    return list

  def close(self, FileHandlers: List[FileHandler]) -> None:
    for fileHandler in FileHandlers:
      fileHandler.close()
    return

  def isChinese(self, word: str) -> bool:
    for ch in word:
      if ('\u4e00' <= ch <= '\u9fff'):
        pass
      else:
        return False
    return True

class LDA:
  '''
    K: 目标主题数
  '''
  def __init__(self) -> None:
    pass

  def sampleData(self, n: int, length: int) -> List[Tuple[str, list[str]]]:
    fileListHanlder = FileListHanlder()
    filePathList = fileListHanlder.getFilePathFromCorpus()
    FileHandlers = fileListHanlder.getFileHandlers(filePathList)
    sample_data = fileListHanlder.sampleFileList(n, length, FileHandlers)
    fileListHanlder.close(FileHandlers)
    return sample_data

  def getAllData(self) -> List[Tuple[str, list[str]]]:
    fileListHanlder = FileListHanlder()
    filePathList = fileListHanlder.getFilePathFromCorpus()
    FileHandlers = fileListHanlder.getFileHandlers(filePathList)
    all_data = fileListHanlder.getAllFile(FileHandlers)
    fileListHanlder.close(FileHandlers)
    return all_data

  def createDictionary(self, data: List[Tuple[str, List[str]]]) -> None:
    data_set = []
    for _, wordList in data:
      data_set.append(wordList)
    self.dictionary = gensim.corpora.Dictionary(data_set)

  def doc2bow(self, data: List[Tuple[str, List[str]]]) -> List[List[Tuple[int, int]]]:
    data_set = []
    for _, wordList in data:
      data_set.append(wordList)
    corpus = [self.dictionary.doc2bow(text) for text in data_set]
    return corpus

  def createLDAModel(self, K: int, corpus: List[List[Tuple[int, int]]]) -> None:
    self.ldaModel = LdaModel(corpus, num_topics= K, id2word=self.dictionary, passes=30, random_state=1, minimum_probability=0)
    all_topic_list = self.ldaModel.print_topics()
    print(all_topic_list)
    return

  def getTopics(self, corpus: List[List[Tuple[int, int]]]) -> List[List[float]]:
    docs_topics_list = []
    for each in self.ldaModel.get_document_topics(corpus):
      res = [probability for _, probability in each]
      docs_topics_list.append(res)
    return docs_topics_list


class SVM:
  def __init__(self) -> None:
    self.sc = StandardScaler()
    self.pca = PCA()
    self.clf = svm.SVC(C=100, kernel='rbf', gamma=0.1, decision_function_shape='ovo')
    return

  def train(self, x_train, y_train) -> None:
    self.sc.fit(x_train)
    x_train_std = self.sc.transform(x_train)

    print('SVM开始训练')
    self.clf.fit(x_train_std, y_train)
    print('SVM训练结束')
    y_train_pred = self.clf.predict(x_train_std)
    report = classification_report(y_true=y_train, y_pred=y_train_pred)
    print('训练集预测结果如下:')
    print(report)
    return

  def predict(self, x, y) -> None:
    x_std = self.sc.transform(x)
    y_pred = self.clf.predict(x_std)
    report = classification_report(y_true=y, y_pred=y_pred)
    print('测试集预测结果如下:')
    print(report)

class App(LDA, SVM):
  def __init__(self, K: int, document_count: int, word_min_size: int) -> None:
    self.K = K
    self.document_count = document_count
    self.word_min_size = word_min_size
    LDA.__init__(self)
    SVM.__init__(self)
    return
  
  def lda(self) -> None:
    '''
    LDA模型训练
    '''
    all_data = self.getAllData()
    self.createDictionary(all_data)
    all_data_bow = self.doc2bow(all_data)
    self.createLDAModel(self.K, all_data_bow)
    ''''''
    sample_data = self.sampleData(self.document_count, self.word_min_size)
    self.clf_data_label = [name for (name, wordList) in sample_data]
    sample_data_bow = self.doc2bow(sample_data)
    self.clf_data = self.getTopics(sample_data_bow)
    return

  def classify(self) -> None:
    x_train, x_test, y_train, y_test = train_test_split(self.clf_data, self.clf_data_label, test_size=0.3, random_state=int(time()))
    self.train(x_train, y_train)
    self.predict(x_test, y_test)
    return
  
  def run(self) -> None:
    self.lda()
    self.classify()
    return

if __name__ == '__main__':
  app = App(K=100, document_count=200, word_min_size=500)
  app.run()