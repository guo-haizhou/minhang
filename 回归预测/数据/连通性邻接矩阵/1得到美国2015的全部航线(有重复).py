import pandas as pd
import numpy as np
import json
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


data = pd.read_csv("D:\Desktop\sou_hangbanshuju\meiguo\拆分csv删除缺失值行/new_bili_meiguo2015.csv")
print(data)
#print(data.shape)            #(694336, 35)

######################################################################################################

c = data.drop_duplicates(subset=['ORIGIN_AIRPORT'], keep='first', inplace=False)
print(len(c))  #出发机场有321个

d = data.drop_duplicates(subset=['DESTINATION_AIRPORT'], keep='first', inplace=False)
print(len(d))  #到达机场也有318个

# #########################################################################################################
# ####打印出共有哪些航线#####################################################
chufa = []
for j in range(len(c['ORIGIN_AIRPORT'].values)):
    aaa = data.loc[data['ORIGIN_AIRPORT'] == c['ORIGIN_AIRPORT'].values[j] ,['ORIGIN_AIRPORT','DESTINATION_AIRPORT']]
    #print(aaa.values)

#得到所有航线的出发机场名称
    for i in range(len(aaa['ORIGIN_AIRPORT'].values)):
        # print(aaa['ORIGIN_AIRPORT'].values[i])
        chufa.append(aaa['ORIGIN_AIRPORT'].values[i])

print(chufa)
print(len(chufa))  # 出发机场的航线总个数

#得到所有航线的到达机场名称
daoda = []
for j in range(len(c['ORIGIN_AIRPORT'].values)):
    aaa = data.loc[data['ORIGIN_AIRPORT'] == c['ORIGIN_AIRPORT'].values[j] ,['ORIGIN_AIRPORT','DESTINATION_AIRPORT']]
    # print(aaa.values)

    for i in range(len(aaa['DESTINATION_AIRPORT'].values)):
        # print(aaa['ORIGIN_AIRPORT'].values[i])
        daoda.append(aaa['DESTINATION_AIRPORT'].values[i])

print(daoda)
print(len(daoda))   #到达机场的航线总个数



dataframe = pd.DataFrame({'ORIGIN_AIRPORT':chufa,'DESTINATION_AIRPORT':daoda})
print(dataframe)

dataframe.to_excel('美国2015全部航线.xlsx', index=False)
