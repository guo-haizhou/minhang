
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler, LabelEncoder,MinMaxScaler
from scipy.optimize import nnls
import warnings
warnings.filterwarnings("ignore")
import re
from sklearn.metrics import accuracy_score
from sklearn import tree,metrics
from imblearn.over_sampling import SMOTE

df = pd.read_csv('D:\Desktop\sou_hangbanshuju\meiguo\拆分csv删除缺失值行/new_bili_meiguo2015.csv',encoding='gbk')
# df = df.dropna(axis=0)
df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# df = df.sample(frac=0.35)
df = df[:113682]

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
print(df[:5])

print(df.isnull().values.any())
print(df.isnull().sum())

df = df[[
         'MONTH','DAY','DAY_OF_WEEK','FLIGHT_NUMBER','SCHEDULED_DEPARTURE','airline_code','chufa_code','air_time','daoda_code',
         'distance',

         "arrive_four"
         ]]


print(df[:5])

#处理数据
columns = [
         'MONTH','DAY','DAY_OF_WEEK','FLIGHT_NUMBER','SCHEDULED_DEPARTURE','airline_code','chufa_code','air_time','daoda_code',
         'distance',

]


for col in columns:
    scaler = MinMaxScaler()
    df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))

print(df[:5])


#构建机器学习模型
from sklearn.model_selection import train_test_split     #训练集：测试集 = 8：2
train_x, test_x, train_y, test_y = train_test_split(df.drop('arrive_four', axis=1), df['arrive_four'], test_size=0.25,random_state=42)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

#重采样前的类别比例
print('重采样前的类别比例:')
print(train_y.value_counts()/len(train_y))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

model = RandomForestClassifier(n_estimators= 150, max_depth=5, min_samples_split=20,
                                min_samples_leaf=2,max_features=10, oob_score=True,random_state=42)

model.fit(train_x,train_y)

# y_predprob4 = model.predict_proba(test_x)      #[:,1]
# string = 'oob_score : %f ,auc : %f' % (model.oob_score_,roc_auc_score(test_y, y_predprob4,multi_class='ovo'))
# #print('oob_score : %f ,auc : %f' % (model.oob_score_,roc_auc_score(test_y, y_predprob4)))
# print('随机森林模型准确率:',re.findall(r"\d+\.?\d*",string)[0])   #利用正则表达式提取小数
# print('auc分数:',re.findall(r"\d+\.?\d*",string)[1])   #利用正则表达式提取小数
# print("\n")


pred = model.predict(test_x)
#模型的预测准确率
print('随机森林模型准确率:',metrics.accuracy_score(test_y,pred))
#模型评价报告
print(metrics.classification_report(test_y,pred))

preddd = model.predict_proba(test_x)
print("随机森林模型ROC值：",roc_auc_score(test_y, preddd,multi_class='ovo'))  #通常拿AUC与0.8比较，如果大于0.8，则认为模型合理