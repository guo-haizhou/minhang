
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  #原先的是3.4.3
import math


plt.figure(figsize=(3,2.5))
# plt.title('各训练集模式下的性能比较')  # 折线图标题
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 显示汉字
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内

config = {
            "font.family": 'serif',
            "font.size": 10.5,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['SimHei'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
plt.rcParams.update(config)


df = pd.read_csv('E:\meiguo2015\将数据集处理成邻接矩阵和特征矩阵\航班数相关性矩阵(101-40)\meiguo2015_hangbanshu40.csv')
print(df)


df = df.iloc[:,:]
print(df)

print(df.columns)

add = []
for i in range(len(df.columns)):
    add.append(df.columns[i])       #round(math.modf(df.columns[i])[1])
print(add)

add2 = []                      #用于将add中的['0.5','0.6']转化成['0','0']
for i in range(len(add)):
    add2.append(round(math.modf(eval(add[i]))[1]))
print(add2)



df_add = pd.DataFrame({0:add2},index=add)
print(df_add)



def insert(df, i, df_add):
    # 指定第i行插入一行数据
    df1 = df.iloc[:i, :]
    df2 = df.iloc[i:, :]
    df_new = pd.concat([df1, df_add, df2], ignore_index=True)
    return df_new

print(insert(df,0,df_add.T))

new_df = insert(df,0,df_add.T)
new_df.columns = new_df.index
print(new_df)
print(type(new_df))

data = pd.DataFrame(new_df,dtype=np.float16)
print(data)

#画热图
# 设置annot=True参数表示显示热图中的数值，annot=False隐藏热图中的数值
cmap = sns.heatmap(data,linewidths = 0.1,cmap=plt.get_cmap('winter_r'), annot=False, annot_kws={"fontsize": 10.5})
plt.yticks(rotation = 0)
plt.xticks(fontsize=10.5)
plt.yticks(fontsize=10.5)
plt.xlabel('机场编码',fontsize=10.5,)
plt.ylabel('机场编码',fontsize=10.5)

cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=10.5)

# #给右边的刻度命名
# cbar = cmap.collections[0].colorbar
# cbar.set_label(label='距离相关性系数',fontsize=7.5)

# plt.savefig("E:\meiguo2015\将数据集处理成邻接矩阵和特征矩阵\距离相关性矩阵/距离邻接矩阵热图402.svg", dpi=3000, format="svg", bbox_inches='tight', pad_inches=0)
plt.savefig("D:\Desktop\研作业\大/航班数邻接矩阵热图.svg", dpi=3000, format="svg", bbox_inches='tight', pad_inches=0)
plt.show()