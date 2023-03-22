

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('E:\meiguo2015\将数据集处理成邻接矩阵和特征矩阵\特征矩阵(先确定满足一定数量的机场)\美国2015处理异常值后的延迟时间特征矩阵.csv')

data = data['ORD']

fig1 = plt.figure(figsize=(5, 3))
plt.plot(data[-100:], 'b-', label='true_time')
plt.legend(loc='best', fontsize=10)
# plt.savefig(path+'/train_rmse.png')
plt.show()