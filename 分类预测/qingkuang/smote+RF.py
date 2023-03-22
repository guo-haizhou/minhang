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
df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# df = df.sample(frac=0.35)
df = df[:116382]

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
print(df[:5])

# print(df.isnull().values.any())
# print(df.isnull().sum())
df = df[[
         #  'MONTH','DAY','DAY_OF_WEEK','FLIGHT_NUMBER','SCHEDULED_DEPARTURE','airline_code','DEPARTURE_TIME','ARRIVAL_TIME',
         # 'SCHEDULED_ARRIVAL', 'DISTANCE',
         'MONTH','DAY','DAY_OF_WEEK','FLIGHT_NUMBER','SCHEDULED_DEPARTURE','airline_code','chufa_code','air_time','daoda_code',
         'distance',

         "arrive_four"
         ]]

print(df[:5])


#处理数据
columns = [
            # 'MONTH', 'DAY', 'DAY_OF_WEEK', 'FLIGHT_NUMBER', 'SCHEDULED_DEPARTURE', 'airline_code', 'DEPARTURE_TIME','ARRIVAL_TIME',
            # 'SCHEDULED_ARRIVAL', 'DISTANCE',
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



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# model = RandomForestClassifier(n_estimators= 250, max_depth=5, min_samples_split=5,
#                                 min_samples_leaf=4,max_features='auto',oob_score=True, random_state=42)

# model = RandomForestClassifier(
# n_estimators=10, criterion='gini', max_depth=None,
# min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
# max_leaf_nodes=None, min_impurity_decrease=0.0,  bootstrap=True, oob_score=False,
# n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None
# )

# model = RandomForestClassifier()
#
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


# #SVM
# clf = svm.SVC(C=0.8,kernel='linear',decision_function_shape='ovr',probability = True) #C:惩罚参数,它表征的是对错误分类的惩罚程度
# #clf = svm.SVC()
# clf.fit(train_x,train_y)
# accuracy1 = clf.score(test_x,test_y)
# print('SVM准确率；',accuracy1)
# probabilities2 = clf.predict_proba(test_x)
# # print(test_Y)
# # print(probabilities2[:, 1])
# roc_auc_score(test_y, probabilities2,multi_class='ovr')
# print('ROC-AUC分数2：',roc_auc_score(test_y, probabilities2,multi_class='ovr'))
# print('\n')
#
#
# #XGBoost
# from sklearn.neighbors import KNeighborsClassifier as XGBoost
# knc = XGBoost(n_neighbors=6 )
# knc.fit(train_x,train_y)
# print('KNN准确率', knc.score(test_x, test_y))
# probabilities3 = knc.predict_proba(test_x)
# # print(test_Y)
# # print(probabilities3[:, 1])
# roc_auc_score(test_y, probabilities3,multi_class='ovr')
# print('ROC-AUC分数3：',roc_auc_score(test_y, probabilities3,multi_class='ovr'))
#
#
#
# #朴素贝叶斯模型
# from sklearn.naive_bayes import GaussianNB
# gnb=GaussianNB()
# gnb.fit(train_x,train_y)
# print('朴素贝叶斯模型准确率：',gnb.score(test_x, test_y))
# prob_pos=gnb.predict_proba(test_x)
# print('ROC-AUC分数4：',roc_auc_score(test_y, prob_pos,multi_class='ovr'))



##############################################################################################################################
####################################################################################################################
#SMOTE采样算法
#利用SMOTE算法对数据进行处理
X_train_smote, y_train_smote = SMOTE().fit_resample(train_x, train_y)


#重采样前的类别比例
print(train_y.value_counts()/len(train_y))
#重采样后的类别比例
print(pd.Series(y_train_smote).value_counts()/len(y_train_smote))



#经过SMOTE算法处理后，两个类别就可以达到1:1的平衡状态
#利用这个平衡状态，重新构建决策树分类器
# clf2 = tree.DecisionTreeClassifier(n_estimators= 50,random_state=42)

# clf2 = RandomForestClassifier(n_estimators= 250,min_samples_split=10,random_state=42)
clf2 = RandomForestClassifier(n_estimators= 200, max_depth=15, min_samples_split=30,
                                min_samples_leaf=2,max_features=10,oob_score=True, random_state=42)

# clf2 = RandomForestClassifier(
# n_estimators=200, criterion='gini', max_depth=30,
# min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
# max_leaf_nodes=None, min_impurity_decrease=0.0,  bootstrap=True, oob_score=False,
# n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None
# )


clf2.fit(X_train_smote, y_train_smote)

pred2 = clf2.predict(np.array(test_x))
#模型的预测准确率
print("SMOTE采样算法处理后的准确率：",metrics.accuracy_score(test_y,pred2))
#模型评价报告
print(metrics.classification_report(test_y,pred2))


pred123 = clf2.predict_proba(test_x)
print("SMOTE处理后的ROC值：",roc_auc_score(test_y, pred123,multi_class='ovo'))


#################################################################################
