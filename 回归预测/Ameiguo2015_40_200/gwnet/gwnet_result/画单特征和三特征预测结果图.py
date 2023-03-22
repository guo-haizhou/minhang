

#本代码是将predicts预测值的三维数组200*325*10，转化成200*325*1的形式，方便求误差
#即先循环将每10个数值求平均值，并追加放在一个列表中，再以每325个数值进行切分，将数据转换为200*325*1
#因为testy真实值的形式就是200*325*1，形式相同才能求误差

import numpy as np

# # 创建数组（3维）
# a = np.arange(100).reshape((10, 5, 2))
# print(a.shape)

# # 存储
#将多维数组保存成npy文件，下面再处理
# np.save(file="data.npy", arr=a)

#单特征预测结果，其实是用的DCRNN网络预测结果
##################################################################################################################################
data1 = np.load('E:\GWNET_daima\Ameiguo2015_40_200\dcrnn\dcrnn_result\meiguo2015_dcrnn_predicts200.npy')
print(data1)
print(data1.shape)


k = data1[49][30]
print(np.mean(k))

m = []       #将200*325个平均值存在m中，后面再以每325个进行切分，重新组成数组
for i in range(200):
    for j in range(40):
       # print(data1[i][j])
       mean = np.mean(data1[i][j])
       m.append(mean)

print(len(m))

#切分m，每隔325个数据切分一次
mm = []
for i in range(0,len(m)-39,40):
    s = m[i:i+40]
    mm.append(s)
    # print(s)

import numpy as np
mm = np.array(mm)
print(mm)
print(mm.shape)


xin_data1 = np.reshape(mm,(200,40,1))
print(xin_data1)
print(xin_data1.shape)

y_preds0 = xin_data1
# y_pred_sample0 = y_preds0[:, 2, 0]

#####################################################################################################################################
#三特征预测结果
data1 = np.load('E:\GWNET_daima\Ameiguo2015_40_200\gwnet\gwnet_result\meiguo2015_gwnet_sange_predicts200.npy')
print(data1)
print(data1.shape)


k = data1[49][30]
print(np.mean(k))

m = []       #将200*325个平均值存在m中，后面再以每325个进行切分，重新组成数组
for i in range(200):
    for j in range(40):
       # print(data1[i][j])
       mean = np.mean(data1[i][j])
       m.append(mean)

print(len(m))

#切分m，每隔325个数据切分一次
mm = []
for i in range(0,len(m)-39,40):
    s = m[i:i+40]
    mm.append(s)
    # print(s)

import numpy as np
mm = np.array(mm)
print(mm)
print(mm.shape)


xin_data1 = np.reshape(mm,(200,40,1))
print(xin_data1)
print(xin_data1.shape)


print("#################################################################################")
####################################################################################
data2 = np.load('E:\GWNET_daima\Ameiguo2015_40_200\dcrnn\dcrnn_result\meiguo2015_dcrnn_testy200.npy')
print(data2)
print(data2.shape)

data2 = np.reshape(data2,(200,40,1))
print(data2)
print(data2.shape)


y_preds = xin_data1     #三特征预测结果
y_true = data2




##############################################################
#画图
"""仅显示一个机场的第一个时间步数据"""
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文问题
plt.rcParams['axes.unicode_minus'] = False #解决负号显示问题
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

config = {
            "font.family": 'serif',
            "font.size": 7.5,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['SimHei'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
plt.rcParams.update(config)


y_sample = y_true[-170:-5, 14, 0]          #即显示的第2列列机场的真实值，可更改
y_pred_sample0 = y_preds0[-170:-5, 14, 0]   #单特征预测值
y_pred_sample = y_preds[-170:-5,14,0]       #多特征预测值

print("真实值：",y_sample)
print("单特征预测值：",y_pred_sample0)
print("三特征预测值：",y_pred_sample)




plt.figure(figsize=(5.1,3.1))
plt.plot(range(len(y_sample)), y_sample, color='blue', linewidth=0.8, label='真实值')
plt.plot(range(len(y_pred_sample0)), y_pred_sample0,color='g', linewidth=0.8, label='单特征情况下的预测值',linestyle='--')
plt.plot(range(len(y_pred_sample)), y_pred_sample,color='red', linewidth=0.8, label='多特征情况下的预测值')   #三特征预测结果
plt.xlabel('时间序列',fontsize=10.5)
plt.ylabel('平均延误时间/minute',fontsize=10.5)

plt.xticks(fontsize=10.5)   # 用星期几替换横坐标x的值
plt.yticks(fontsize=10.5)

plt.legend(loc="upper right",prop={'family' : 'SimHei', 'size': 10.5})
#plt.savefig("E:\GWNET_daima\Ameiguo2015_40_200\gwnet\gwnet_result/单_多特征对比图.svg", dpi=600, format="svg", bbox_inches='tight', pad_inches=0)
# plt.savefig("E:\meiguo2015\一\图/单_多特征对比图.svg", dpi=3000, format="svg", bbox_inches='tight', pad_inches=0)
plt.savefig("D:\Desktop\研作业\大/单_多特征对比图2.svg", dpi=3000, format="svg", bbox_inches='tight', pad_inches=0)
plt.show()







