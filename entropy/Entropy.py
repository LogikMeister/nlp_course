from io import TextIOWrapper

'''Entropy Demo
  1. Calculate the information entropy of Chinese by using the database.
'''

import jieba
import math
import os

class ReadLine:
  '''
  Class: Readline - 按self.endCharList中的所有char分割文本的Generator
  '''
  MAX_SIZE = 100
  endCharList = ['.', '。', '?', '？', ';', '；', '...']

  def __init__(self, f: TextIOWrapper) -> None:
    self.buf = ''
    self.f = f
    return
  
  def __iter__(self):
    return self

  def __next__(self):
    pos, char = self.search()
    if pos >= 0 and char:
      line = self.buf[: pos]
      self.buf = self.buf[pos + len(char):]
      return line
    else:
      chunk = self.f.read(self.MAX_SIZE)
      if chunk:
        self.buf += chunk
        return self.__next__()
      else:
        self.f.close()
        raise StopIteration

  def search(self):
    dict = {}
    for char in self.endCharList:
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

class NodeTree:
  def __init__(self, word: str = 'root', isRoot: bool = True, level: int = 0) -> None:
    self.nextNodeDict = {}
    self.nextCount = 0
    self.count = 1
    self.parentNode = None
    self.word = word
    self.level = level
    if isRoot:
      self.levelDict = {}
      self.levelCount = {}
    return

  def add(self, wordList: list[str], index: int = 0) -> None:
    if len(wordList) <= index:
      return
    word: str = wordList[index]
    nextNode: NodeTree = self.nextNodeDict.get(word)
    if nextNode == None:
      nextNode = NodeTree(word = word, isRoot = False, level = self.level + 1)
      nextNode.parentNode = self
      self.nextNodeDict[word] = nextNode
      nextNode.bindToRoot(nextNode)
    else:
      nextNode.count += 1
    self.nextCount += 1
    self.countUpdateToRoot(nextNode)
    nextNode.add(wordList, index + 1)
    return

  def bindToRoot(self, node) -> None:
    parentNode: NodeTree = self.parentNode
    bindNode: NodeTree = node
    if parentNode:
      parentNode.bindToRoot(node)
    else:
      nodeList: list = self.levelDict.get(bindNode.level)
      if nodeList != None:
        nodeList.append(bindNode)
      else:
        self.levelDict[bindNode.level] = [bindNode]
    return

  def countUpdateToRoot(self, node) -> None:
    parentNode: NodeTree = self.parentNode
    updateNode: NodeTree = node
    if parentNode:
      parentNode.countUpdateToRoot(node)
    else:
      prevCount = self.levelCount.get(updateNode.level)
      if prevCount != None:
        self.levelCount[updateNode.level] += 1
      else:
        self.levelCount[updateNode.level] = 1
      return

  def clear(self) -> None:
    self.nextNodeDict.clear()
    self.nextCount = 0
    return


class Entropy:
  def __init__(self) -> None:
    self.nodeTree = NodeTree()
    self.charCount = 0
    self.wordCount = 0

  # 工具函数

  def getFilePathFromDB(self) -> list[str]:
    '''
    Function: getFilePathFromDB - 返回.\database中所有txt文件地址列表
    '''
    list = []
    dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')
    for filename in os.listdir(dirname):
      list.append(os.path.join(dirname, filename))
    return list

  def readFile(self, filePath: str) -> ReadLine:
    '''
    Function: readFile - 读取TXT

    Parameters:
      filePath: str - txt文件地址

    Return:
      lines: Iterator - 返回文本文件的句子迭代器
    '''
    file = open(filePath, encoding='utf-8')
    lines = ReadLine(file)
    return lines

  def isChinese(self, word: str) -> bool:
    '''
    Function: isChinese - 判断是否所有字符均为中文

    Parameters:
      word: str - 被判断的字符串

    Return: bool - 所有字符均为汉字返回True, 否则为False
    '''
    for ch in word:
      if ('\u4e00' <= ch <= '\u9fff'):
        pass
      else:
        return False
    return True


  # 核心函数
  def divide(self, line: str):
    '''
    Function: divide - 分词处理返回分词结果列表

    Parameters:
      line: str - 输入字符串分词处理

    Return:
      wordList: list[str] - 返回分词字符串列表
    '''
    wordList = []
    for word in jieba.cut(line, cut_all=False):
      if self.isChinese(word):
        wordList.append(word)
    return wordList

  def updateDict(self, wordList: list[str], mode: int = 1) -> None:
    '''
    Function: updateDict - 根据分词结果更新词语之间的关系NodeTree, 并记录字数与分词个数

    Parameters:
      wordList: list[str] - 分词结果
      mode: int - 以mode作为最高元，进行NodeTree更新

    Return: None
    '''
    self.wordCount += len(wordList)
    charCount = 0
    for word in wordList:
      charCount += len(word)
    self.charCount += charCount
    for i in range(0, len(wordList)):
      list = wordList[i: i + mode]
      self.nodeTree.add(list)
    return

  def clear(self) -> None:
    '''
    Function: clear- 初始化NodeTree, 汉字和分词个数
    Return: None
    '''
    self.nodeTree.clear()
    self.charCount = 0
    self.wordCount = 0
    return

  def calculate(self, mode: int = 1) -> float:
    '''
    Function: calculate - 根据词语之间的关系计算信息熵

    Parameters:
      mode: int = 1 - 计算N元模型信息熵

    Return:
      h: float - 信息熵
    '''
    nodeList: list[NodeTree] = self.nodeTree.levelDict.get(mode)
    if not nodeList:
      return
    h: float = 0
    for node in nodeList:
      parentNode: NodeTree = node.parentNode
      h -= node.count / self.nodeTree.levelCount[mode] * math.log2( node.count / parentNode.nextCount )
    return h

  def run(self, mode: int = 1) -> None:
    '''
    Function: run - 默认运行，统计.\database中所有txt文件, 并计算信息熵

    Parameters:
      mode: int = 1 - 以mode作为最高元, 计算所有模型的信息熵
                      eg. mode = 3, 计算一元、二元、三元模型的信息熵
    
    Return: - None
    '''
    self.clear()
    filePathList = self.getFilePathFromDB()
    for filePath in filePathList:
      print(filePath)
      lines = self.readFile(filePath)
      for line in lines:
        wordList = self.divide(line)
        self.updateDict(wordList, mode)
    print('计算信息熵所用数据库字数总计%d字分词个数为%d词, 平均词长为%f' %(self.charCount, self.wordCount, self.charCount / self.wordCount))
    for i in range(0, mode):
      print('%d元模型信息熵: %f比特/词' %(i+1, app.calculate(i + 1)))
    return

if __name__ == '__main__':
  app = Entropy()
  app.run(mode=3)