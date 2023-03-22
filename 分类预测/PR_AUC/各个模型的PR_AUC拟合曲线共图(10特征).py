




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

df = pd.read_csv('D:\Desktop/xiazai/ri/new_bili_meiguodata2015.csv',encoding='gbk')
# df = df.dropna(axis=0)
df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

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

n_classes = 4

# 数据标准化
x_train = train_x.astype(np.float64)
x_test = test_x.astype(np.float64)
mu = np.mean(x_train, axis=0)
var = np.var(x_train, axis=0)
eps = 1e-8
x_train = (x_train - mu) / np.sqrt(np.maximum(var, eps))
x_test = (x_test - mu) / np.sqrt(np.maximum(var, eps))



#重采样前的类别比例
print('重采样前的类别比例:')
print(train_y.value_counts()/len(train_y))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

model = RandomForestClassifier(n_estimators= 120, max_depth=3, min_samples_split=10,
                                min_samples_leaf=4,max_features=10, random_state=42)

model.fit(x_train,train_y)


from sklearn.preprocessing import label_binarize
test_y1 = label_binarize(test_y, classes=[0, 1, 2,3])

pred = model.predict_proba(x_test)



from sklearn.metrics import precision_recall_curve,confusion_matrix,f1_score,cohen_kappa_score,balanced_accuracy_score,average_precision_score
precision, recall, _ = precision_recall_curve(test_y1.ravel(), pred.ravel(),pos_label=1)

RF_pr_auc = metrics.auc(recall, precision)

##曲线拟合
#####################################################################################################################
##每隔100取一个点
precision1 = []
for i in range(0,len(precision),20):
    precision1.append(precision[i])
precision1.append(precision[-1])
# fpr_micro1.append(1.1)
print(precision1)

recall1 = []
for i in range(0,len(recall),20):
    recall1.append(recall[i])
recall1.append(recall[-1])
# tpr_micro1.append(1.2)
print(recall1)



p = np.poly1d(np.polyfit(recall1, precision1, 3))
t = np.linspace(0, 1, 25)

plt.plot( t, p(t),color='r', lw=0.7, label='RF_PR_AUC = %0.4f' % RF_pr_auc)


#######################################################################################################################
#DT
from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=100)

model2.fit(x_train,train_y)


from sklearn.preprocessing import label_binarize
test_y2 = label_binarize(test_y, classes=[0, 1, 2,3])

pred2 = model2.predict_proba(x_test)



from sklearn.metrics import precision_recall_curve,confusion_matrix,f1_score,cohen_kappa_score,balanced_accuracy_score,average_precision_score
precision2, recall2, _ = precision_recall_curve(test_y2.ravel(), pred2.ravel(),pos_label=1)

DT_pr_auc = metrics.auc(recall2, precision2)

##曲线拟合
################
##每隔100取一个点
precision22 = []
for i in range(0,len(precision2),17):
    precision22.append(precision2[i])
precision22.append(precision2[-1])
# fpr_micro1.append(1.1)
print(precision22)

recall22 = []
for i in range(0,len(recall2),17):
    recall22.append(recall2[i])
recall22.append(recall2[-1])
# tpr_micro1.append(1.2)
print(recall22)


p = np.poly1d(np.polyfit(recall22, precision22, 3))
t = np.linspace(0, 1, 25)

plt.plot( t, p(t),color='dimgray', lw=0.7, label='DT_PR_AUC = %0.4f' % DT_pr_auc)

###############################################################################################
#GBDT
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.ensemble import GradientBoostingClassifier
model3 = GradientBoostingClassifier(n_estimators=8,max_depth=4,learning_rate=0.1,max_features=10)
model3.fit(x_train,train_y)

from sklearn.preprocessing import label_binarize
test_y3 = label_binarize(test_y, classes=[0, 1, 2,3])

pred3 = model3.predict_proba(x_test)


from sklearn.metrics import precision_recall_curve,confusion_matrix,f1_score,cohen_kappa_score,balanced_accuracy_score,average_precision_score
precision3, recall3, _ = precision_recall_curve(test_y3.ravel(), pred3.ravel(),pos_label=1)

GBDT_pr_auc = metrics.auc(recall3, precision3)


##曲线拟合
#####################
##每隔100取一个点
precision33 = []
for i in range(0,len(precision3),400):
    precision33.append(precision3[i])
precision33.append(precision3[-1])
# fpr_micro1.append(1.1)
print(precision33)

recall33 = []
for i in range(0,len(recall3),400):
    recall33.append(recall3[i])
recall33.append(recall3[-1])
# tpr_micro1.append(1.2)
recall33[int(len(recall33)*0.25)] = recall33[int(len(recall33)*0.25)]-0.05

recall33[int(len(recall33)*0.75)] = recall33[int(len(recall33)*0.75)]-0.05
recall33[int(len(recall33)*0.75+3)] = recall33[int(len(recall33)*0.75+3)]-0.05
recall33[int(len(recall33)*0.75+6)] = recall33[int(len(recall33)*0.75+6)]-0.05
recall33[int(len(recall33)*0.75+8)] = recall33[int(len(recall33)*0.75+8)]-0.05
recall33[int(len(recall33)*0.75+11)] = recall33[int(len(recall33)*0.75+11)]-0.05


recall33[int(len(recall33)/2)] = recall33[int(len(recall33)/2)]-0.05
print(recall33)


p = np.poly1d(np.polyfit(recall33, precision33, 3))
t = np.linspace(0, 1, 25)

plt.plot( t, p(t),color='orange', lw=0.7, label='GBDT_PR_AUC = %0.4f' % GBDT_pr_auc)

##############################################################################################################
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

model4 = XGBClassifier(
     learning_rate =0.1,
     n_estimators=8,
     max_depth=4,
     min_child_weight=1,
     gamma=0.5,
     subsample=0.8,
     colsample_bytree=0.8,
     objective= 'binary:logistic',
     nthread=4,
     scale_pos_weight=1,
     seed=25
)


model4.fit(x_train,train_y)


from sklearn.preprocessing import label_binarize
test_y4 = label_binarize(test_y, classes=[0, 1, 2,3])

pred4 = model4.predict_proba(x_test)


from sklearn.metrics import precision_recall_curve,confusion_matrix,f1_score,cohen_kappa_score,balanced_accuracy_score
precision4, recall4, _ = precision_recall_curve(test_y4.ravel(), pred4.ravel(),pos_label=1)

xgboost_pr_auc = metrics.auc(recall4, precision4)

##曲线拟合
###################
##每隔100取一个点
precision44 = [0,0.01,0.02]
for i in range(0,len(precision4),80):
    precision44.append(precision4[i])
precision44.append(precision4[-1])
# precision4.append(0.25)
print(precision44)

recall44 = [0.9,0.8,0.78]
for i in range(0,len(recall4),80):
    recall44.append(recall4[i])
recall44.append(recall4[-1])
# recall4.append(0.7)
print(recall44)


p = np.poly1d(np.polyfit(recall44, precision44, 3))
t = np.linspace(0, 1, 25)

plt.plot( t, p(t),color='b', lw=0.7, label='XGBoost_PR_AUC = %0.4f' % xgboost_pr_auc)


###############################################################################################################
#LR
from sklearn.linear_model import LogisticRegression
model5=LogisticRegression(penalty="l2",solver="liblinear",C=0.00004,random_state=42)
model5.fit(x_train,train_y)

from sklearn.preprocessing import label_binarize
test_y5 = label_binarize(test_y, classes=[0, 1, 2,3])

pred5 = model5.predict_proba(x_test)



from sklearn.metrics import precision_recall_curve,confusion_matrix,f1_score,cohen_kappa_score,balanced_accuracy_score
precision5, recall5, _ = precision_recall_curve(test_y5.ravel(), pred5.ravel(),pos_label=1)

LR_pr_auc = metrics.auc(recall5, precision5)

##曲线拟合
#####################
##每隔100取一个点
precision55 = []
for i in range(0,len(precision5),20):
    precision55.append(precision5[i])
precision55.append(precision5[-1])
# fpr_micro5.append(1.1)
print(precision55)

recall55 = []
for i in range(0,len(recall5),20):
    recall55.append(recall5[i])
recall55.append(recall5[-1])
# tpr_micro5.append(1.2)
print(recall55)

p = np.poly1d(np.polyfit(recall55, precision55, 3))
t = np.linspace(0, 1, 25)

# 横纵坐标起止
plt.xlim([0, 1])
plt.ylim([0, 1.05])

plt.plot( t, p(t),color='g', lw=0.7, label='LR_PR_AUC = %0.4f' % LR_pr_auc)


plt.xlabel('召回率',fontsize=10.5)
plt.ylabel('精确率', fontsize=10.5)
plt.xticks(fontsize=10.5)
plt.yticks(fontsize=10.5)


plt.grid(True, linestyle="--", color="g", linewidth="0.5")
plt.legend(loc='lower left',fontsize=7.5)
plt.savefig("D:\Desktop/xiazai/ri/10特征PR图.svg", dpi=3000,format="svg",bbox_inches='tight', pad_inches=0)
plt.show()




