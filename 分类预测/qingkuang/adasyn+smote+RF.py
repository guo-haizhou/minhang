


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

df = pd.read_csv('D:\Desktop/xiazai/ri/new_bili_meiguodata2015.csv',encoding='gbk')
df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
print(df[:5])


print(df.isnull().values.any())
print(df.isnull().sum())
df = df[[
         # 'MONTH','DAY','DAY_OF_WEEK','FLIGHT_NUMBER','SCHEDULED_DEPARTURE','DEPARTURE_TIME','airline_code','chufa_code','daoda_code',
         # 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME','AIR_TIME','DISTANCE',
         # # 'MONTH','DAY','DAY_OF_WEEK','FLIGHT_NUMBER','SCHEDULED_DEPARTURE','DEPARTURE_TIME',,
         # # 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME','AIR_TIME','DISTANCE',
         'MONTH','DAY','DAY_OF_WEEK','FLIGHT_NUMBER','SCHEDULED_DEPARTURE','airline_code','daoda_code','air_time','chufa_code',
         'distance',

         "arrive_four"
         ]]

print(df[:5])


#处理数据
columns = [
        # 'MONTH','DAY','DAY_OF_WEEK','FLIGHT_NUMBER','SCHEDULED_DEPARTURE','DEPARTURE_TIME','airline_code','chufa_code','daoda_code',
        #  'SCHEDULED_ARRIVAL','AIR_TIME','DISTANCE',
        # # 'MONTH','DAY','DAY_OF_WEEK','FLIGHT_NUMBER','SCHEDULED_DEPARTURE','DEPARTURE_TIME','daoda_code',
        # # 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME','AIR_TIME','DISTANCE',
         'MONTH','DAY','DAY_OF_WEEK','FLIGHT_NUMBER','SCHEDULED_DEPARTURE','airline_code','daoda_code','air_time','chufa_code',
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



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# model = RandomForestClassifier(n_estimators= 250, max_depth=9, min_samples_split=10,
#                                 min_samples_leaf=5,max_features='auto',oob_score=True, random_state=10)
#
# model.fit(train_x,train_y)
#
# # y_predprob4 = model.predict_proba(test_x)      #[:,1]
# # string = 'oob_score : %f ,auc : %f' % (model.oob_score_,roc_auc_score(test_y, y_predprob4,multi_class='ovo'))
# # #print('oob_score : %f ,auc : %f' % (model.oob_score_,roc_auc_score(test_y, y_predprob4)))
# # print('随机森林模型准确率:',re.findall(r"\d+\.?\d*",string)[0])   #利用正则表达式提取小数
# # print('auc分数:',re.findall(r"\d+\.?\d*",string)[1])   #利用正则表达式提取小数
# # print("\n")
#
#
# pred = model.predict(test_x)
# #模型的预测准确率
# print('随机森林模型准确率:',metrics.accuracy_score(test_y,pred))
# #模型评价报告
# print(metrics.classification_report(test_y,pred))
#
# preddd = model.predict_proba(test_x)
# print("随机森林模型ROC值：",roc_auc_score(test_y, preddd,multi_class='ovo'))  #通常拿AUC与0.8比较，如果大于0.8，则认为模型合理



#可视化模型的输出
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#绘制ROC曲线
plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签
plt.rcParams['axes.unicode_minus']=False


##############################################################################################################################
####################################################################################################################
print("#################################################################")
print("ADASYN处理后的结果：")
from imblearn.over_sampling import ADASYN
ada = ADASYN(n_neighbors=2,random_state=42)
X_res, y_res = ada.fit_resample(df.drop('arrive_four', axis=1), df['arrive_four'])


#重采样前的类别比例
print(train_y.value_counts()/len(train_y))
#重采样后的类别比例
print(pd.Series(y_res).value_counts()/len(y_res))



#SMOTE采样算法
#利用SMOTE算法对数据进行处理
# over_samples = SMOTE(random_state=10)
over_samples = SMOTE(random_state=42)
over_samples_X,over_samples_y = over_samples.fit_resample(X_res, y_res)

print(over_samples_X.shape)
print(over_samples_y.shape)


#重采样前的类别比例
# print(train_y.value_counts()/len(train_y))
#重采样后的类别比例
print(pd.Series(over_samples_y).value_counts()/len(over_samples_y))



#经过SMOTE算法处理后，两个类别就可以达到1:1的平衡状态

#利用这个平衡状态，重新构建决策树分类器
# clf2 = tree.DecisionTreeClassifier()
# clf2 = RandomForestClassifier()
clf2 = RandomForestClassifier(n_estimators= 200,max_depth=16, min_samples_split=20,
                                min_samples_leaf=2,max_features=10,oob_score=True, random_state=42)
# n_estimators= 200, max_depth=1, min_samples_split=13,min_samples_leaf=5,max_features='auto,oob_score=True, random_state=42



clf2.fit(over_samples_X,over_samples_y)

pred2 = clf2.predict(np.array(test_x))
#模型的预测准确率
print("ADASYN和SMOTE采样算法处理后的准确率：",metrics.accuracy_score(test_y,pred2))
#模型评价报告
print(metrics.classification_report(test_y,pred2))


y_predprob4 = clf2.predict_proba(test_x)      #[:,1]
string = 'oob_score : %f ,auc : %f' % (clf2.oob_score_,roc_auc_score(test_y, y_predprob4,multi_class='ovo'))
#print('oob_score : %f ,auc : %f' % (model.oob_score_,roc_auc_score(test_y, y_predprob4)))
print('随机森林模型准确率:',re.findall(r"\d+\.?\d*",string)[0])   #利用正则表达式提取小数


pred123 = clf2.predict_proba(test_x)
print("ADASYN和SMOTE处理后的ROC值：",roc_auc_score(test_y, pred123,multi_class='ovo'))


#################################################################################

