
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy.linalg as la
import math
from sklearn.svm import SVR
from statsmodels.tsa.arima_model import ARIMA


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    data1 = np.mat(data)
    train_size = int(time_len * rate)
    train_data = data1[0:train_size]
    test_data = data1[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])
    return trainX, trainY, testX, testY


###### evaluation ######
def evaluation(a, b):
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    MAPE = np.mean(np.abs((a - b) / a)) * 100
    F_norm = la.norm(a - b) / la.norm(a)
    r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    var = 1 - (np.var(a - b)) / np.var(a)
    return rmse, mae, MAPE, 1 - F_norm, r2, var


# path = r'D:\Desktop\github\T-GCN-master\data\los_speed.csv'
path = r'E:\GWNET_daima\Ameiguo2015_40_200\meiguo_data\重新生成的美国2015延误时间特征矩阵.csv'

data = pd.read_csv(path)

time_len = data.shape[0]
num_nodes = data.shape[1]

print(time_len)
print(num_nodes)

train_rate = 0.946
seq_len = 15
pre_len = 3
trainX, trainY, testX, testY = preprocess_data(data, time_len, train_rate, seq_len, pre_len)
method = 'HA'  ####HA or SVR or ARIMA


print(len(testX))
print(len(testY))


########### HA 历史平均模型#############
if method == 'HA':
    result = []
    for i in range(len(testX)):
        a = np.array(testX[i])
        tempResult = []

        a1 = np.mean(a, axis=0)
        tempResult.append(a1)
        a = a[1:]
        a = np.append(a, [a1], axis=0)
        a1 = np.mean(a, axis=0)
        tempResult.append(a1)
        a = a[1:]
        a = np.append(a, [a1], axis=0)
        a1 = np.mean(a, axis=0)
        tempResult.append(a1)

        result.append(tempResult)
    result1 = np.array(result)
    result1 = np.reshape(result1, [-1, num_nodes])
    testY1 = np.array(testY)
    testY1 = np.reshape(testY1, [-1, num_nodes])

    print(result1[:10])
    print(testY1[:10])

    rmse, mae, MAPE, accuracy, r2, var = evaluation(testY1, result1)


    print(testY1.shape)
    print(result1.shape)

    print('HA_rmse:%r' % rmse,
          'HA_mae:%r' % mae,
          'HA_mape:%r' % MAPE,
          'HA_acc:%r' % accuracy,
          'HA_r2:%r' % r2,
          'HA_var:%r' % var)