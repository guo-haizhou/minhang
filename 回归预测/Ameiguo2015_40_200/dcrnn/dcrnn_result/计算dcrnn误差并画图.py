

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



data1 = np.load('E:\GWNET_daima\Ameiguo2015_40_200\dcrnn\dcrnn_result\meiguo2015_dcrnn_predicts200.npy')
print(data1)
print(data1.shape)


k = data1[99][39]
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


y_preds = xin_data1
y_true = data2




##############################################################
#画图
"""仅显示一个机场的第一个时间步数据"""
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文问题
plt.rcParams['axes.unicode_minus'] = False #解决负号显示问题

y_sample = y_true[:, 2, 0]          #即显示的第2列列机场的真实值，可更改
y_pred_sample = y_preds[:, 2, 0]

print("真实值：",y_sample)
print("预测值：",y_pred_sample)


#求MAPE时，分母不能为0，所以查看下某列的真实值中是否有0
y_sample = list(y_sample)
if 0 in y_sample:
    print("查看真实值中0所在的索引位置：",y_sample.index(0))
    print("验证此位置是否是0：",y_sample[y_sample.index(0)])

    y_pred_sample = list(y_pred_sample)
    print("查看预测值中对应索引位置的值是几：",y_pred_sample[y_sample.index(0)])


plt.figure(figsize=(10,7))
plt.plot(range(len(y_sample)), y_sample, color='blue', linewidth=2.5, label='Ground Truth')
plt.plot(range(len(y_pred_sample)), y_pred_sample,color='red', linewidth=2.5, label='Predictions')
plt.xlabel('时间序列')
plt.ylabel('延误时间')
plt.legend(loc="upper left")
plt.show()





##############################################################
#计算三维数据预测值和三维数据真实值的误差
"""整体评估测试数据的预测性能"""
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true +=  1e-18 #add small values to true velocities to avoid division by zeros
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print('MAE: ', mean_absolute_error(y_true.flatten(), y_preds.flatten()))
print('RMSE: ', np.sqrt(mean_squared_error(y_true.flatten(), y_preds.flatten())))
#print('MAPE: ', mean_absolute_percentage_error(y_true.flatten(), y_preds.flatten()), "%")






