
import pandas as pd
t = pd.read_csv('E:\meiguo2015\将数据集处理成邻接矩阵和特征矩阵\特征矩阵(先确定满足一定数量的机场)\meiguo2015_yanwu_time(有大量异常值).csv')

t1 = t[abs(t - t.mean()) > 3*t.std()].dropna(how = 'all')   #含有异常值的行
t1_1 = t[abs(t - t.mean()) <= 3*t.std()].dropna(how = 'all')   #不含有异常值的行
#print(t1.index)
# print(t1_1.index)
# print(len(t1_1))
# print("含有异常值的行，需删除：",len(t1))
print(t1)


import numpy as np

for i in range(len(list(t1_1))):
    t1_1[list(t1_1)[i]].replace(np.nan, round(np.mean(t1_1[list(t1_1)[i]])),inplace = True)  #将t1_1的LIT列中为空值的地方，替换为此列的平均值
print(t1_1[:20])


#第二遍检验异常值，并处理
t1_2 = t1_1[abs(t1_1 - t1_1.mean()) <= 3*t1_1.std()].dropna(how = 'all')
print(t1_2[:20])


for i in range(len(list(t1_2))):
    t1_2[list(t1_2)[i]].replace(np.nan, round(np.mean(t1_2[list(t1_2)[i]])),inplace = True)  #将t1_1的LIT列中为空值的地方，替换为此列的平均值
print(t1_2[:20])


#第三遍检验异常值，并处理
t1_3 = t1_2[abs(t1_2 - t1_2.mean()) <= 3*t1_2.std()].dropna(how = 'all')
print(t1_3[:20])


for i in range(len(list(t1_3))):
    t1_3[list(t1_3)[i]].replace(np.nan, round(np.mean(t1_3[list(t1_3)[i]])),inplace = True)  #将t1_1的LIT列中为空值的地方，替换为此列的平均值
print(t1_3[:20])


#第四遍检验异常值，并处理
t1_4 = t1_3[abs(t1_3 - t1_3.mean()) <= 3*t1_3.std()].dropna(how = 'all')
print(t1_4[:20])

for i in range(len(list(t1_4))):
    t1_4[list(t1_4)[i]].replace(np.nan, round(np.mean(t1_4[list(t1_4)[i]])),inplace = True)  #将t1_1的LIT列中为空值的地方，替换为此列的平均值
print(t1_4[:20])


#第五遍检验异常值，并处理
t1_5 = t1_4[abs(t1_4 - t1_4.mean()) <= 3*t1_4.std()].dropna(how = 'all')
print(t1_5[:20])

for i in range(len(list(t1_5))):
    t1_5[list(t1_5)[i]].replace(np.nan, round(np.mean(t1_5[list(t1_5)[i]])),inplace = True)  #将t1_1的LIT列中为空值的地方，替换为此列的平均值
print(t1_5[:20])

#第六遍检验异常值，并处理
t1_6 = t1_5[abs(t1_5 - t1_5.mean()) <= 3*t1_5.std()].dropna(how = 'all')
print(t1_6[:20])

for i in range(len(list(t1_6))):
    t1_6[list(t1_6)[i]].replace(np.nan, round(np.mean(t1_6[list(t1_6)[i]])),inplace = True)  #将t1_1的LIT列中为空值的地方，替换为此列的平均值
print(t1_6[:20])


#第七遍检验异常值，并处理
t1_7 = t1_6[abs(t1_6 - t1_6.mean()) <= 3*t1_6.std()].dropna(how = 'all')
print(t1_7[:20])

for i in range(len(list(t1_7))):
    t1_7[list(t1_7)[i]].replace(np.nan, round(np.mean(t1_7[list(t1_7)[i]])),inplace = True)  #将t1_1的LIT列中为空值的地方，替换为此列的平均值
print(t1_7)

#第八遍检验异常值，并处理
t1_8 = t1_7[abs(t1_7 - t1_7.mean()) <= 3*t1_7.std()].dropna(how = 'all')
print(t1_8[:20])

for i in range(len(list(t1_8))):
    t1_8[list(t1_8)[i]].replace(np.nan, round(np.mean(t1_8[list(t1_8)[i]])),inplace = True)  #将t1_1的LIT列中为空值的地方，替换为此列的平均值
print(t1_8[:20])


#第九遍检验异常值，并处理
t1_9 = t1_8[abs(t1_8 - t1_8.mean()) <= 3*t1_8.std()].dropna(how = 'all')
print(t1_9[:20])

for i in range(len(list(t1_9))):
    t1_9[list(t1_9)[i]].replace(np.nan, round(np.mean(t1_9[list(t1_9)[i]])),inplace = True)  #将t1_1的LIT列中为空值的地方，替换为此列的平均值
print(t1_9)



#检查最终数据中是否有nan
print(t1_9.isnull().values.any())

###################################################################################################
t1_9.to_csv('美国2015处理异常值后的延迟时间特征矩阵.csv',index = False)