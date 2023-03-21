
import pandas as pd
import numpy as np
import torch

pt_version = torch.__version__
print(pt_version)



def read_data(features_csv, adj_csv):
    urban_core_speeds_chuli = pd.read_csv(features_csv)
    urban_core_speeds = urban_core_speeds_chuli.T
    print(urban_core_speeds)


    adj_matrix_df = pd.read_csv(adj_csv, header=None)
    adj_matrix = np.array(adj_matrix_df)



    speeds_df = urban_core_speeds.iloc[:,:2000]       #从第7列往后的列才是需要的真实值。

    speeds = np.array(speeds_df)

    return adj_matrix_df, urban_core_speeds, adj_matrix, speeds



adj_matrix_df, urban_core_speeds, adj_matrix, speeds = read_data('E:\GWNET_daima\Ameiguo2015_40_200\meiguo_data\重新生成的美国2015延误时间特征矩阵.csv',
                                                                 'E:\GWNET_daima\Ameiguo2015_40_200\meiguo_data\meiguo2015_liantongxing40.csv')

"""划分训练集和测试集以及验证集，2500/250 = 10，则划分为10份，训练集占8份，测试集占1份，验证集占1份"""
test_split = 210 # 6 days X 12 measurements per hour X 24 hours
val_split = 210

split = speeds.shape[1] - test_split
train_data = speeds[:, :split-int(val_split)]
test_data = speeds[:, split:]
val_data = speeds[:, split-int(val_split):split]

print("Train data: ", train_data.shape)      #(304, 6336)
print("Test data: ", test_data.shape)        #(304, 1728)
print("Val data: ", val_data.shape)          #(304, 576)



"""归一化数据"""
mean_speed = train_data.mean()
std_speed = train_data.std()

train_data = (train_data - mean_speed) / (std_speed)
test_data = (test_data - mean_speed) / (std_speed)
val_data = (val_data - mean_speed) / (std_speed)

print(train_data.shape)        #(304, 6336)
print(test_data.shape)         #(304, 1728)
print(val_data.shape)          #(304, 576)


#GCN_LSTM例子
"""Ref: https://stellargraph.readthedocs.io/en/stable/demos/time-series/gcn-lstm-time-series.html"""

sequence_len = 10
prediction_len = 1

def build_features_labels(sequence_len, prediction_len, train_data, test_data, val_data):
    X_train, Y_train, X_test, Y_test, X_val, Y_val = [], [], [], [], [], []

    for i in range(train_data.shape[1] - int(sequence_len + prediction_len - 1)):
        a = train_data[:, i: i + sequence_len + prediction_len]
        X_train.append(a[:, :sequence_len])
        Y_train.append(a[:, sequence_len:sequence_len + prediction_len])

    for i in range(test_data.shape[1] - int(sequence_len + prediction_len - 1)):
        b = test_data[:, i: i + sequence_len + prediction_len]
        X_test.append(b[:, :sequence_len])
        Y_test.append(b[:, sequence_len:sequence_len + prediction_len])

    for i in range(val_data.shape[1] - int(sequence_len + prediction_len - 1)):
        b = val_data[:, i: i + sequence_len + prediction_len]
        X_val.append(b[:, :sequence_len])
        Y_val.append(b[:, sequence_len:sequence_len + prediction_len])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)

    return X_train, Y_train, X_test, Y_test, X_val, Y_val


X_train, Y_train, X_test, Y_test, X_val, Y_val = build_features_labels(sequence_len,
                                                                       prediction_len,
                                                                       train_data,
                                                                       test_data,
                                                                       val_data)

print(X_train.shape)             #(6322, 304, 10)        横向滑窗，因为csv中数据横向表示时序，10表示用前10列值预测后面紧跟的5个值
print(Y_train.shape)             #(6322, 304, 5)
print(X_test.shape)              #(1714, 304, 10)
print(Y_test.shape)              #(1714, 304, 5)
print(X_val.shape)               #(562, 304, 10)
print(Y_val.shape)               #(562, 304, 5)




edges = np.nonzero(adj_matrix)                 #用于得到数组array中非零元素的位置
edges = np.vstack([edges, adj_matrix[edges]])    #垂直(行)按顺序堆叠数组
edge_index = edges[:2, :].astype(float)
edge_attr = edges[2, :].astype(float)
print('Edges shape: ', edge_index.shape, ', Attr shape: ',edge_attr.shape)


from torch_geometric_temporal.signal import StaticGraphTemporalSignal               #时间图神经网络扩展库
train_loader = StaticGraphTemporalSignal(edge_index, edge_attr, X_train, Y_train)
test_loader = StaticGraphTemporalSignal(edge_index, edge_attr, X_test, Y_test)
val_loader = StaticGraphTemporalSignal(edge_index, edge_attr, X_val, Y_val)

next(iter(train_loader))

print(train_loader[0])      #Data(x=[228, 10], edge_index=[2, 2889216], edge_attr=[2889216], y=[228, 5])

##############################################################################################



#####################################################################################################


import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN

class DCRNNModel(torch.nn.Module):
    def __init__(self, node_features, output_len):
        super(DCRNNModel, self).__init__()
        self.dcrnn = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, output_len)

    def forward(self, x, edge_index, edge_weight):
        h = self.dcrnn(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


def evaluate_model(model, val_loader):
    loss = 0
    step = 0
    model.eval()
    with torch.no_grad():
        for snapshot in val_loader:
            snapshot = snapshot.to(device)
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            loss = loss + torch.mean(torch.abs(y_hat - snapshot.y))
            step += 1
        loss = loss / (step + 1)

    print("Val MAE: {:.4f}".format(loss.item()))
    return loss


model = DCRNNModel(node_features=10, output_len=1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
device = torch.device("cuda")


train_losses, val_losses = [], []


for epoch in range(45):
    print('Epoch: ', epoch + 1)
    print('==========')
    loss = 0

    model.train().to(device)
    for time, snapshot in enumerate(train_loader):
        snapshot = snapshot.to(device)
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss = loss + torch.mean(torch.abs(y_hat - snapshot.y))
    loss = loss / (time + 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    train_losses.append(loss.item())

    print("Train MAE: {:.4f}".format(loss.item()))

    va_loss = evaluate_model(model, val_loader)
    val_losses.append(va_loss.item())



#画图
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.plot(range(1, len(train_losses)+1), train_losses, color='blue', linewidth=2.5,label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, color='red', linewidth=2.5, label='Val Loss')
plt.legend(loc="upper left")
plt.show()

#####################################################################################
"""预测测试集数据"""
model.eval()
y_preds = list()
y_true = list()


model.train().to(device)          #需要加此行代码，即将模型在GPU上跑，不然下面代码会出错。
for snapshot in test_loader:
    snapshot = snapshot.to(device)
    y = snapshot.y.cpu().numpy()         #如果数据是被送到GPU上的话，那么转换numpy前一定要转换到cpu。
    y_pred = model(snapshot.x, snapshot.edge_index,snapshot.edge_attr).view(len(snapshot.x), -1).cpu().detach().numpy()



    y = np.array((y * std_speed) + mean_speed)
    y_pred = np.array((y_pred * std_speed) + mean_speed)
    y_preds.extend(list(y_pred))
    y_true.extend(list(y))

y_preds = np.array(y_preds)
y_true = np.array(y_true)
y_preds = y_preds.reshape(int(y_preds.shape[0] / (40)), 40, 1)
y_true = y_true.reshape(int(y_true.shape[0] / (40)), 40, 1)

print(y_true.shape)                 #(1714, 325, 5)
print(y_preds.shape)                #(1714, 325, 5)


print("数据类型：",type(y_true))


########################################################################################
#将预测值和真实值保存在npy文件中，方便画图
baocun_predicts = y_preds
np.save(file="E:\GWNET_daima\Ameiguo2015_40_200\dcrnn\dcrnn_result/meiguo2015_dcrnn_predicts200.npy", arr=baocun_predicts)

baocun_testy = y_true
np.save(file="E:\GWNET_daima\Ameiguo2015_40_200\dcrnn\dcrnn_result/meiguo2015_dcrnn_testy200.npy", arr=baocun_testy)
########################################################################################

"""仅显示一个机场的第一个时间步数据"""
plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文问题
plt.rcParams['axes.unicode_minus'] = False #解决负号显示问题

y_sample = y_true[:, 5, 0]          #即显示的第三列机场的真实值，可更改
y_pred_sample = y_preds[:, 5, 0]

print("真实值：",y_sample)
print("预测值：",y_pred_sample)

plt.figure(figsize=(10,7))
plt.plot(range(len(y_sample)), y_sample, color='blue', linewidth=2.5, label='Ground Truth')
plt.plot(range(len(y_pred_sample)), y_pred_sample,color='red', linewidth=2.5, label='Predictions')
plt.xlabel('时间序列')
plt.ylabel('延误时间')
plt.legend(loc="upper left")
plt.show()



"""整体评估测试数据的预测性能"""
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true +=  1e-18 #add small values to true velocities to avoid division by zeros
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print('MAE: ', mean_absolute_error(y_true.flatten(), y_preds.flatten()))
print('RMSE: ', np.sqrt(mean_squared_error(y_true.flatten(), y_preds.flatten())))
# print('MAPE: ', mean_absolute_percentage_error(y_true.flatten(), y_preds.flatten()), "%")


