
import pandas as pd
import numpy as np
from collections import Counter

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

data = pd.read_csv('D:\Desktop\sou_hangbanshuju\meiguo\拆分csv删除缺失值行/new_bili_meiguo2015.csv')
print(data)

chufa = data['ORIGIN_AIRPORT'].values
print(len(chufa))

#统计出发机场中各个三字码的数量分别有多少
c = Counter(chufa)
dict = dict(c)
print("各个出发机场以及对应的航班数量：")
print(dict)
print("\n")

#将机场以及数量所形成的字典，按照值的大小降序排列
print("按照各机场数量的大小降序排列：")
print(sorted(dict.items(), key=lambda x: x[1],reverse=True)[:40])  #观察发现有59个机场的数量超过2000条

cunchu_jichang = []
m = sorted(dict.items(), key=lambda x: x[1],reverse=True)[:40]
for i in range(len(m)):
    cunchu_jichang.append(m[i][0])

print(cunchu_jichang)
print("航班数量超过2000的机场有",len(cunchu_jichang),"个")   #40个，因此创建的邻接矩阵为40*40，特征矩阵为2000*40

print(",".join(cunchu_jichang))

