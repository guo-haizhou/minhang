

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('E:\meiguo2015\将数据集处理成邻接矩阵和特征矩阵\特征矩阵(先确定满足一定数量的机场)\重新生成的美国2015延误时间特征矩阵.csv')


#画列图
df_column = data['0']

fig1 = plt.figure(figsize=(5, 3))
plt.plot(df_column[-100:], 'b-', label='true_time')
plt.legend(loc='best', fontsize=10)
# plt.savefig(path+'/train_rmse.png')
plt.show()

#画行图  最后一行
df_hang = data.iloc[-1,:]
# print(df_hang)

fig2 = plt.figure(figsize=(5, 3))
plt.plot(df_hang, 'b-', label='true_time')
plt.legend(loc='best', fontsize=0.1)
# plt.savefig(path+'/train_rmse.png')
plt.show()
