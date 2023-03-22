

# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from scipy.optimize import nnls
import warnings

warnings.filterwarnings("ignore")
import re
from sklearn.metrics import accuracy_score
from sklearn import tree, metrics
from imblearn.over_sampling import SMOTE
import time


plt.figure(figsize=(3.1,2.4))
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内




config = {
            "font.family": 'serif',
            "font.size": 7.5,# 相当于小四大小
            "mathtext.fontset": 'stix',#matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
            "font.serif": ['Microsoft YaHei'],#宋体
            'axes.unicode_minus': False # 处理负号，即-号
         }
plt.rcParams.update(config)

start = time.time()

df = pd.read_csv('D:\Desktop/xiazai/ri/new_bili_meiguodata2015.csv', encoding='gbk')
# df = df.dropna(axis=0)
df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
print(df)

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

print(df[:5])
print(df.isnull().values.any())
print(df.isnull().sum())

#################################################################################################################


df = df[[
         'MONTH','DAY','DAY_OF_WEEK','FLIGHT_NUMBER','SCHEDULED_DEPARTURE','airline_code','chufa_code','air_time','daoda_code','distance',

         'chufa_tianqi','chufa_fengxiang','chufa_zuigao_wendu','chufa_zuidi_wendu','chufa_fengli',
         'daoda_tianqi','daoda_fengxiang','daoda_zuigao_wendu','daoda_zuidi_wendu','daoda_fengli',

         "arrive_four"

         ]]

print(df[:5])


# 构建机器学习模型
from sklearn.model_selection import train_test_split  # 训练集：测试集 = 8：2

train_x, test_x, train_y, test_y = train_test_split(df.drop('arrive_four', axis=1), df['arrive_four'], test_size=0.25,
                                                    random_state=42)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

from sklearn.preprocessing import label_binarize
test_y1 = label_binarize(test_y, classes=[0, 1, 2,3])

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,roc_curve,auc

model = RandomForestClassifier(n_estimators=90,
                               max_depth=10,
                               min_samples_split=20, min_samples_leaf=4, random_state=42)

model.fit(train_x, train_y)

pre_score = model.predict_proba(test_x)

fpr_micro,tpr_micro,_ = roc_curve(test_y1.ravel(),pre_score.ravel())
print(fpr_micro)
print(tpr_micro)

auc = auc(fpr_micro,tpr_micro)
print(auc)

# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_micro, tpr_micro,"r",linewidth = 3)
# plt.xlabel("假正率")
# plt.ylabel("真正率")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.grid()
# plt.title("随机森林ROC曲线")
# plt.text(0.2,0.8,"AUC = "+str(round(auc,4)))
# plt.show()

######################################################################################################################
#每隔100取一个点
fpr_micro1 = []
for i in range(0,len(fpr_micro),2000):
    fpr_micro1.append(fpr_micro[i])
fpr_micro1.append(fpr_micro[-1])
fpr_micro1.append(1.1)
print(fpr_micro1)

tpr_micro1 = []
for i in range(0,len(tpr_micro),2000):
    tpr_micro1.append(tpr_micro[i])
tpr_micro1.append(tpr_micro[-1])
tpr_micro1.append(1.2)
print(tpr_micro1)
######################################################################################################################
#第一次拟合曲线
import os
import numpy as np
# from scipy import log
from scipy.optimize import curve_fit
import math
from sklearn.metrics import r2_score
# 字体

# 拟合函数
def func(x, a, b):
    #    y = a * log(x) + b
    y = x/ (a * x + b)
    return y


# 拟合的坐标点
x0 = fpr_micro1
y0 = tpr_micro1

# 拟合，可选择不同的method
result = curve_fit(func, x0, y0, method='trf')
a, b = result[0]

# 绘制拟合曲线用
x1 = np.arange(0, 1, 0.0001)
# y1 = a * log(x1) + b
y1 = x1 / (a * x1 + b)

x0 = np.array(x0)
y0 = np.array(y0)
# 计算r2
y2 = x0 / (a * x0 + b)
# y2 = a * log(x0) + b
r2 = r2_score(y0, y2)

# plt.figure(figsize=(7.5, 5))
# 坐标字体大小
# plt.tick_params(labelsize=7.5)
# 原数据散点
# plt.scatter(x0, y0, s=30, marker='o')

# 横纵坐标起止
# plt.xlim((0, 1))
# plt.ylim((0, round(max(y0))+0.05))

# 拟合曲线
plt.plot(x1, y1, color='r', lw=0.7,label='RF_AUC = %0.4f' % auc)
# plt.xlabel('False Positive Rate', fontsize=7.5)
# plt.ylabel('True Positive Rate', fontsize=7.5)
# plt.title('Some extension of Receiver operating characteristic to multi-class', fontsize=7.5)

# True 显示网格
# linestyle 设置线显示的类型(一共四种)
# color 设置网格的颜色
# linewidth 设置网格的宽度
# plt.grid(True, linestyle="--", color="g", linewidth="0.5")
# plt.show()

######################################################################################################33

from sklearn.metrics import roc_auc_score,roc_curve,auc
#DT
from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier(criterion='entropy',max_depth=13,max_features=20,random_state=100)


model2.fit(train_x, train_y)

pre_score2 = model2.predict_proba(test_x)

fpr_micro2,tpr_micro2,_ = roc_curve(test_y1.ravel(),pre_score2.ravel())
print(fpr_micro2)
print(tpr_micro2)

auc2 = auc(fpr_micro2,tpr_micro2)
print(auc2)

# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_micro2, tpr_micro2,"r",linewidth = 3)
# plt.xlabel("假正率")
# plt.ylabel("真正率")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.grid()
# plt.title("DT的ROC曲线")
# plt.text(0.2,0.8,"AUC = "+str(round(auc2,4)))
# plt.show()

####################################
#每隔100取一个点
fpr_micro11 = []
for i in range(0,len(fpr_micro2),20):
    fpr_micro11.append(fpr_micro2[i])
fpr_micro11.append(fpr_micro2[-1])
fpr_micro11.append(1.1)
print(fpr_micro11)

tpr_micro11 = []
for i in range(0,len(tpr_micro2),20):
    tpr_micro11.append(tpr_micro2[i])
tpr_micro11.append(tpr_micro2[-1])
tpr_micro11.append(1.2)
print(tpr_micro11)
#########################
#第一次拟合曲线
import os
import numpy as np
# from scipy import log
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score
# 字体
plt.rcParams['font.sans-serif' ] =['SimHei']

# 拟合函数
def func(x, a, b):
    #    y = a * log(x) + b
    y = x/ (a * x + b)
    return y


# 拟合的坐标点
x0 = fpr_micro11
y0 = tpr_micro11

# 拟合，可选择不同的method
result = curve_fit(func, x0, y0, method='trf')
a, b = result[0]

# 绘制拟合曲线用
x1 = np.arange(0, 1, 0.0001)
# y1 = a * log(x1) + b
y1 = x1 / (a * x1 + b)

x0 = np.array(x0)
y0 = np.array(y0)
# 计算r2
y2 = x0 / (a * x0 + b)
# y2 = a * log(x0) + b
r2 = r2_score(y0, y2)

# plt.figure(figsize=(7.5, 5))
# 坐标字体大小
# plt.tick_params(labelsize=7.5)
# 原数据散点
# plt.scatter(x0, y0, s=30, marker='o')

# 横纵坐标起止
# plt.xlim((0, 1))
# plt.ylim((0, round(max(y0))+0.05))

# 拟合曲线
plt.plot(x1, y1, color='dimgray', lw=0.7,label='DT_AUC = %0.4f' % auc2)
# plt.xlabel('False Positive Rate', fontsize=7.5)
# plt.ylabel('True Positive Rate', fontsize=7.5)
# plt.title('Some extension of Receiver operating characteristic to multi-class', fontsize=7.5)

# True 显示网格
# linestyle 设置线显示的类型(一共四种)
# color 设置网格的颜色
# linewidth 设置网格的宽度
# plt.grid(True, linestyle="--", color="g", linewidth="0.5")
# plt.legend(loc='lower right')
# plt.show()

######################################################################################################################3
#GBDT
from sklearn.ensemble import GradientBoostingClassifier
model3 = GradientBoostingClassifier(n_estimators=10,max_depth=7,max_features=20)

model3.fit(train_x, train_y)

pre_score3 = model3.predict_proba(test_x)

fpr_micro3,tpr_micro3,_ = roc_curve(test_y1.ravel(),pre_score3.ravel())
print(fpr_micro3)
print(tpr_micro3)

auc3 = auc(fpr_micro3,tpr_micro3)
print(auc3)

# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_micro, tpr_micro,"r",linewidth = 3)
# plt.xlabel("假正率")
# plt.ylabel("真正率")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.grid()
# plt.title("GBDT的ROC曲线")
# plt.text(0.2,0.8,"AUC = "+str(round(auc,4)))
# plt.show()

###################
#每隔100取一个点
fpr_micro111 = []
for i in range(0,len(fpr_micro3),2000):
    fpr_micro111.append(fpr_micro3[i])
fpr_micro111.append(fpr_micro3[-1])
fpr_micro111.append(1.1)
print(fpr_micro111)

tpr_micro111 = []
for i in range(0,len(tpr_micro3),2000):
    tpr_micro111.append(tpr_micro3[i])
tpr_micro111.append(tpr_micro3[-1])
tpr_micro111.append(1.2)
print(tpr_micro111)
########################
#第一次拟合曲线
import os
import numpy as np
# from scipy import log
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# 拟合函数
def func(x, a, b):
    #    y = a * log(x) + b
    y = x/ (a * x + b)
    return y


# 拟合的坐标点
x0 = fpr_micro111
y0 = tpr_micro111

# 拟合，可选择不同的method
result = curve_fit(func, x0, y0, method='trf')
a, b = result[0]

# 绘制拟合曲线用
x1 = np.arange(0, 1, 0.0001)
# y1 = a * log(x1) + b
y1 = x1 / (a * x1 + b)

x0 = np.array(x0)
y0 = np.array(y0)
# 计算r2
y2 = x0 / (a * x0 + b)
# y2 = a * log(x0) + b
r2 = r2_score(y0, y2)

# plt.figure(figsize=(7.5, 5))
# 坐标字体大小
# plt.tick_params(labelsize=7.5)
# 原数据散点
# plt.scatter(x0, y0, s=30, marker='o')

# 横纵坐标起止
# plt.xlim((0, 1))
# plt.ylim((0, round(max(y0))+0.05))

# 拟合曲线
plt.plot(x1, y1, color='orange', lw=0.7,label='GBDT_AUC = %0.4f' % auc3)
# plt.xlabel('False Positive Rate', fontsize=7.5)
# plt.ylabel('True Positive Rate', fontsize=7.5)
# plt.title('Some extension of Receiver operating characteristic to multi-class', fontsize=7.5)

# True 显示网格
# linestyle 设置线显示的类型(一共四种)
# color 设置网格的颜色
# linewidth 设置网格的宽度
# plt.grid(True, linestyle="--", color="g", linewidth="0.5")
# plt.legend(loc='lower right')
# plt.show()

###################################################################################################################
#XGBoost
from xgboost import XGBClassifier

model = XGBClassifier(
     learning_rate =0.1,
     n_estimators=30,
     max_depth=7,
     min_child_weight=1,
     gamma=0.5,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=4,
     scale_pos_weight=1,
     seed=30
)

model.fit(train_x, train_y)

pre_score4 = model.predict_proba(test_x)

fpr_micro4,tpr_micro4,_ = roc_curve(test_y1.ravel(),pre_score4.ravel())
print(fpr_micro4)
print(tpr_micro4)

auc4 = auc(fpr_micro4,tpr_micro4)
print(auc4)

# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_micro, tpr_micro,"r",linewidth = 3)
# plt.xlabel("假正率")
# plt.ylabel("真正率")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.grid()
# plt.title("XGBoost的ROC曲线")
# plt.text(0.2,0.8,"AUC = "+str(round(auc,4)))
# plt.show()

####################
#每隔100取一个点
fpr_micro1111 = []
for i in range(0,len(fpr_micro4),800):
    fpr_micro1111.append(fpr_micro4[i])
fpr_micro1111.append(fpr_micro4[-1])
fpr_micro1111.append(1.15)
print(fpr_micro1111)

tpr_micro1111 = []
for i in range(0,len(tpr_micro4),800):
    tpr_micro1111.append(tpr_micro4[i])
tpr_micro1111.append(tpr_micro4[-1])
tpr_micro1111.append(1.25)
print(tpr_micro1111)
######################################################################################################################
#第一次拟合曲线
import os
import numpy as np
# from scipy import log
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# 拟合函数
def func(x, a, b):
    #    y = a * log(x) + b
    y = x/ (a * x + b)
    return y


# 拟合的坐标点
x0 = fpr_micro1111
y0 = tpr_micro1111

# 拟合，可选择不同的method
result = curve_fit(func, x0, y0, method='trf')
a, b = result[0]

# 绘制拟合曲线用
x1 = np.arange(0, 1, 0.0001)
# y1 = a * log(x1) + b
y1 = x1 / (a * x1 + b)

x0 = np.array(x0)
y0 = np.array(y0)
# 计算r2
y2 = x0 / (a * x0 + b)
# y2 = a * log(x0) + b
r2 = r2_score(y0, y2)

# plt.figure(figsize=(7.5, 5))
# 坐标字体大小
# plt.tick_params(labelsize=7.5)
# 原数据散点
# plt.scatter(x0, y0, s=30, marker='o')

# 横纵坐标起止
# plt.xlim((0, 1))
# plt.ylim((0, round(max(y0))+0.05))

# 拟合曲线
plt.plot(x1, y1, color='b', lw=0.7,label='XGBoost_AUC = %0.4f' % auc4)
# plt.xlabel('False Positive Rate', fontsize=7.5)
# plt.ylabel('True Positive Rate', fontsize=7.5)
# plt.title('Some extension of Receiver operating characteristic to multi-class', fontsize=7.5)

# True 显示网格
# linestyle 设置线显示的类型(一共四种)
# color 设置网格的颜色
# linewidth 设置网格的宽度
# plt.grid(True, linestyle="--", color="g", linewidth="0.5")
# plt.legend(loc='lower right')
# plt.show()

################################################################################################################
#LR
from sklearn.linear_model import LogisticRegression
model5=LogisticRegression(penalty="l2",solver="liblinear",tol=0.01)  #=0.00000001

model5.fit(train_x, train_y)

pre_score5 = model5.predict_proba(test_x)

fpr_micro5,tpr_micro5,_ = roc_curve(test_y1.ravel(),pre_score5.ravel())
print(fpr_micro5)
print(tpr_micro5)

auc5 = auc(fpr_micro5,tpr_micro5)
print(auc5)

# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_micro5, tpr_micro5,"r",linewidth = 3)
# plt.xlabel("假正率")
# plt.ylabel("真正率")
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.grid()
# plt.title("随机森林ROC曲线")
# plt.text(0.2,0.8,"AUC = "+str(round(auc,4)))
# plt.show()

##############################
#每隔100取一个点
fpr_micro11111 = []
for i in range(0,len(fpr_micro5),1200):
    fpr_micro11111.append(fpr_micro5[i])
fpr_micro11111.append(fpr_micro5[-1])
fpr_micro11111.append(1.15)
print(fpr_micro11111)

tpr_micro11111 = []
for i in range(0,len(tpr_micro5),1200):
    tpr_micro11111.append(tpr_micro5[i])
tpr_micro11111.append(tpr_micro5[-1])
tpr_micro11111.append(1.25)
print(tpr_micro11111)
#########################
#第一次拟合曲线
import os
import numpy as np
# from scipy import log
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# 拟合函数
def func(x, a, b):
    #    y = a * log(x) + b
    y = x/ (a * x + b)
    return y


# 拟合的坐标点
x0 = fpr_micro11111
y0 = tpr_micro11111

# 拟合，可选择不同的method
result = curve_fit(func, x0, y0, method='trf')
a, b = result[0]

# 绘制拟合曲线用
x1 = np.arange(0, 1, 0.0001)
# y1 = a * log(x1) + b
y1 = x1 / (a * x1 + b)

x0 = np.array(x0)
y0 = np.array(y0)
# 计算r2
y2 = x0 / (a * x0 + b)
# y2 = a * log(x0) + b
r2 = r2_score(y0, y2)

# plt.figure(figsize=(7.5, 5))
# 坐标字体大小
# plt.tick_params(labelsize=7.5)
# 原数据散点
# plt.scatter(x0, y0, s=30, marker='o')

# 横纵坐标起止
plt.xlim([0, 1])
plt.ylim([0, 1.05])

# 拟合曲线
plt.plot(x1, y1, color='g', lw=0.7,label='LR_AUC = %0.4f' % auc5)

plt.xlabel('假正率',fontsize=10.5)
plt.ylabel('真正率', fontsize=10.5)

plt.xticks(fontsize=10.5)
plt.yticks(fontsize=10.5)

# True 显示网格
# linestyle 设置线显示的类型(一共四种)
# color 设置网格的颜色
# linewidth 设置网格的宽度
plt.grid(True, linestyle="--", color="g", linewidth="0.5")
plt.legend(loc='lower right',fontsize=7.5)
plt.savefig("D:\Desktop/xiazai/ri/20特征ROC图.svg", dpi=3000,format="svg",bbox_inches='tight', pad_inches=0)
plt.show()


