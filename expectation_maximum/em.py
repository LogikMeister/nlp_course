from typing import Dict, List

'''
一个袋子中三种硬币的混合比例为s1, s2与1-s1-s2 (0<=si<=1)
三种硬币掷出正面的概率分别为: p, q, r
1. 自己指定系数s1, s2, p, q, r. 生成N个投掷硬币的结果(由01构成的序列, 其中1为正面,0为反面) 
2. 利用EM算法来对参数进行估计并与预先假定的参数进行比较
'''

import msvcrt
import random

class CoinBox:

  def __init__(self, s1: float, s2: float, n: int) -> None:
    if s1 + s2 > 1:
      raise Exception('s1 + s2 > 1')
    elif s1 > 1 or s1 < 0:
      raise Exception('s1 > 1 or s1 < 0')
    elif s2 > 1 or s2 < 0:
      raise Exception('s2 > 1 or s2 < 0')
    else:
      self.s1 = s1
      self.s2 = s2
      self.s3: float = 1 - self.s1 - self.s2
      self.generator(n)
    return

  def generator(self, n: int) -> None:
    self.count = n
    count_a = int(self.s1 * n)
    count_b = int(self.s2 * n)
    count_c = n - count_a - count_b
    self.coin_list = []
    for _ in range(0, count_a):
      self.coin_list.append('A')
    for _ in range(0, count_b):
      self.coin_list.append('B')
    for _ in range(0, count_c):
      self.coin_list.append('C')
    return
  
  def getACoin(self) -> str:
    random_index = int(random.random() * len(self.coin_list))
    if len(self.coin_list) == 0:
      raise Exception('coin box is empty')
    else:
      return self.coin_list[random_index]

class CoinBoxHandler:
  def __init__(self, opts: Dict[str, float], n: int) -> None:
    keyList = ['s1', 's2', 'p', 'q', 'r']

    for key in keyList:
      if key not in opts.keys():
        raise Exception('opts don\'t has key: %s' %(key))
      else:
        if not isinstance(opts[key], float):
          raise Exception('opts.%s is not float' %(key))

    self.p: float = opts['p']
    self.q: float = opts['q']
    self.r: float = opts['r']
    self.s1: float = opts['s1']
    self.s2: float = opts['s2']
    self.coin_box = CoinBox(self.s1, self.s2, n)

    return

  def getAResult(self, coin: str, toss_count: int) -> List[int]:
    list = []
    for _ in range(toss_count):
      random_num = random.random()
      if coin == 'A':
        if random_num > self.p:
          list.append(0)
        else:
          list.append(1)
      elif coin == 'B':
        if random_num > self.q:
          list.append(0)
        else:
          list.append(1)
      elif coin == 'C':
        if random_num > self.r:
          list.append(0)
        else:
          list.append(1)
    return list

  def getResultList(self, test_count: int, toss_count: int) -> List[int]:
    list = []
    for _ in range(test_count):
      coin = self.coin_box.getACoin()
      result = self.getAResult(coin, toss_count)
      obcerse_count = 0
      for each in result:
        obcerse_count += each
      list.append(obcerse_count)
    return list
  
class EM:
  def __init__(self, opts: Dict[str, float]) -> None:
    self.opts = opts
    coin_count = 1000000
    self.coinBoxHandler = CoinBoxHandler(self.opts, coin_count)
    return

  def solveAttrs(self, test_count: int, toss_count: int, initAttrs: Dict[str, float]) -> None:
    initAttrsKeyList = ['s1', 's2', 'p', 'q', 'r']

    for key in initAttrsKeyList:
      if key not in initAttrs.keys():
        raise Exception('initAttrs don\'t has key: %s' %(key))
      else:
        if not isinstance(initAttrs[key], float):
          raise Exception('initAttrs.%s is not float' %(key))

    res = self.coinBoxHandler.getResultList(test_count, toss_count)
    s1 = initAttrs['s1']
    s2 = initAttrs['s2']
    p = initAttrs['p']
    q = initAttrs['q']
    r = initAttrs['r']
    CASE_A_1 = [0 for _ in range(test_count)]
    CASE_A_0 = [0 for _ in range(test_count)]
    CASE_B_1 = [0 for _ in range(test_count)]
    CASE_B_0 = [0 for _ in range(test_count)]
    CASE_C_1 = [0 for _ in range(test_count)]
    CASE_C_0 = [0 for _ in range(test_count)]
    
    iterations = 1
    while(True):
      new_s1 = 0
      new_s2 = 0
      for i in range(test_count):
        P_A = (p)**res[i]*(1-p)**(toss_count-res[i]) * s1
        P_B = (q)**res[i]*(1-q)**(toss_count-res[i]) * s2
        P_C = (r)**res[i]*(1-r)**(toss_count-res[i]) * (1 - s1 - s2)
        Q_A = P_A / (P_A + P_B + P_C)
        Q_B = P_B / (P_A + P_B + P_C)
        Q_C = 1 - Q_A - Q_B
        CASE_A_1[i] = res[i] * Q_A
        CASE_A_0[i] = (toss_count-res[i]) * Q_A
        CASE_B_1[i] = res[i] * Q_B
        CASE_B_0[i] = (toss_count-res[i]) * Q_B
        CASE_C_1[i] = res[i] * Q_C
        CASE_C_0[i] = (toss_count-res[i]) * Q_C
        new_s1 += Q_A
        new_s2 += Q_B
      new_s1 /= test_count
      new_s2 /= test_count
      new_p = sum(CASE_A_1) / (sum(CASE_A_1) + sum(CASE_A_0))
      new_q = sum(CASE_B_1) / (sum(CASE_B_1) + sum(CASE_B_0))
      new_r = sum(CASE_C_1) / (sum(CASE_C_1) + sum(CASE_C_0))
      print('第%d次迭代, s1:%.16f, s2:%.16f, p:%.16f, q: %.16f, r: %.16f' %(iterations, new_s1, new_s2, new_p, new_q, new_r))
      if (abs(new_p - p) < 1e-15 and abs(new_q - q) < 1e-15 and abs(new_r - r) < 1e-15 and abs(s1 - new_s1) < 1e-15 and abs(s2 - new_s2) < 1e-15) or msvcrt.kbhit():
        print('迭代结束')
        print('本次实验共实验次数: %d. 每个硬币抛出次数为: %d. 总迭代次数为: %d' %(test_count, toss_count, iterations))
        print('估计参数为: s1:%.16f, s2:%.16f, p:%.16f, q: %.16f, r: %.16f' %(new_s1, new_s2, new_p, new_q, new_r))
        print('实际参数为: s1:%.16f, s2:%.16f, p:%.16f, q: %.16f, r: %.16f' %(self.opts['s1'], self.opts['s2'], self.opts['p'], self.opts['q'], self.opts['r']))
        break
      else:
        s1, s2, p, q, r = new_s1, new_s2, new_p, new_q, new_r
        iterations += 1


if __name__ == '__main__':
  demo = EM(opts={'s1': 0.1, 's2': 0.3, 'p': 0.25, 'q': 0.35, 'r': 0.75})
  demo.solveAttrs(test_count=1000, toss_count=100, initAttrs={'s1': 0.2, 's2': 0.5, 'p': 0.1, 'q': 0.3, 'r': 0.5})

  