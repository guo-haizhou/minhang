

import pandas as pd
import numpy as np
from collections import Counter

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

data = pd.read_csv('D:\Desktop\sou_hangbanshuju\meiguo\拆分csv删除缺失值行/new_bili_meiguo2015.csv')
print(data)

chufa = data['ORIGIN_AIRPORT'].values.tolist()
print(chufa)
print(len(chufa))

orign = ['ORD', 'DFW', 'LAX', 'ATL', 'DEN', 'SFO', 'IAH', 'PHX', 'LAS', 'SEA', 'BOS', 'LGA', 'JFK', 'MSP',
         'EWR', 'DTW', 'MCO', 'CLT', 'BWI', 'PHL', 'SLC', 'MIA', 'SAN', 'MDW', 'DCA', 'FLL', 'DAL', 'HOU',
         'TPA', 'BNA', 'STL', 'PDX', 'OAK', 'AUS', 'CLE', 'SMF', 'SJC', 'SNA', 'MCI', 'IAD']

shuzu = []
for i in range(0,327364,1636):
    a = chufa[i:i+1636]
    print(a)
    jichang1 = a.count('ORD')
    jichang2 = a.count('DFW')
    jichang3 = a.count('LAX')
    jichang4 = a.count('ATL')
    jichang5 = a.count('DEN')
    jichang6 = a.count('SFO')
    jichang7 = a.count('IAH')
    jichang8 = a.count('PHX')
    jichang9 = a.count('LAS')
    jichang10 = a.count('SEA')
    jichang11 = a.count('BOS')
    jichang12 = a.count('LGA')
    jichang13 = a.count('JFK')
    jichang14 = a.count('MSP')
    jichang15 = a.count('EWR')
    jichang16 = a.count('DTW')
    jichang17 = a.count('MCO')
    jichang18 = a.count('CLT')
    jichang19 = a.count('BWI')
    jichang20 = a.count('PHL')
    jichang21 = a.count('SLC')
    jichang22 = a.count('MIA')
    jichang23 = a.count('SAN')
    jichang24 = a.count('MDW')
    jichang25 = a.count('DCA')
    jichang26 = a.count('FLL')
    jichang27 = a.count('DAL')
    jichang28 = a.count('HOU')
    jichang29 = a.count('TPA')
    jichang30 = a.count('BNA')
    jichang31 = a.count('STL')
    jichang32 = a.count('PDX')
    jichang33 = a.count('OAK')
    jichang34 = a.count('AUS')
    jichang35 = a.count('CLE')
    jichang36 = a.count('SMF')
    jichang37 = a.count('SJC')
    jichang38 = a.count('SNA')
    jichang39 = a.count('MCI')
    jichang40 = a.count('IAD')
    hang = [jichang1, jichang2, jichang3,jichang4, jichang5, jichang6,jichang7, jichang8, jichang9,jichang10,
          jichang11, jichang12, jichang13,jichang14, jichang15, jichang16,jichang17, jichang18, jichang19,jichang20,
          jichang21, jichang22, jichang23,jichang24, jichang25, jichang26,jichang27, jichang28, jichang29,jichang30,
          jichang31, jichang32, jichang33,jichang34, jichang35, jichang36,jichang37, jichang38, jichang39,jichang40]
    shuzu.append(hang)
print(shuzu)

df = pd.DataFrame(shuzu)
print(df)


