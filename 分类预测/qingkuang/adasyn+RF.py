

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
df = df[:116382]

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
print(df[:5])

# print(df.isnull().values.any())
# print(df.isnull().sum())
df = df[[
         # 'MONTH','DAY','DAY_OF_WEEK','FLIGHT_NUMBER','SCHEDULED_DEPARTURE','DEPARTURE_TIME','daoda_code',
         # 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME','AIR_TIME','DISTANCE',
         'MONTH','DAY','DAY_OF_WEEK','FLIGHT_NUMBER','SCHEDULED_DEPARTURE','airline_code','chufa_code','air_time','daoda_code',
         'distance',


         "arrive_four"
         ]]


print(df[:5])


#处理数据
columns = [
         #  'MONTH','DAY','DAY_OF_WEEK','FLIGHT_NUMBER','SCHEDULED_DEPARTURE','DEPARTURE_TIME','daoda_code',
         # 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME','AIR_TIME','DISTANCE',
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

print("#################################################################")
print("ADASYN处理后的结果：")
from imblearn.over_sampling import ADASYN
ada = ADASYN(random_state=42)
X_res, y_res = ada.fit_resample(train_x,train_y)

#重采样前的类别比例
print(train_y.value_counts()/len(train_y))
#重采样后的类别比例
print(pd.Series(y_res).value_counts()/len(y_res))



#经过SMOTE算法处理后，两个类别就可以达到1:1的平衡状态
#利用这个平衡状态，重新构建决策树分类器
# clf2 = tree.DecisionTreeClassifier()
# clf2 = RandomForestClassifier(random_state=42)
clf2 = RandomForestClassifier(n_estimators= 200, max_depth=15, min_samples_split=30,
                                min_samples_leaf=1,max_features=10,oob_score=True, random_state=42)

clf2.fit(X_res,y_res)


pred2 = clf2.predict(np.array(test_x))
#模型的预测准确率
print("ADASYN采样算法处理后的准确率：",metrics.accuracy_score(test_y,pred2))
#模型评价报告
print(metrics.classification_report(test_y,pred2))


pred123 = clf2.predict_proba(test_x)
print("ADASYN处理后的ROC值：",roc_auc_score(test_y, pred123,multi_class='ovo'))


#################################################################################

#可视化混淆矩阵
######################################################################################
from sklearn.metrics import confusion_matrix
y_predicted789 = clf2.predict(test_x)

print(test_y)
print(y_predicted789)

confusion_matrix(test_y, y_predicted789)         #y_predicted, test_y
print(confusion_matrix(y_predicted789, test_y, labels=[0,1,2,3]))   #其中predicted = model.predict(test_x)

# #可视化
import numpy as np
def plot_matrix(y_predicted789, test_y, labels_name, title=None,thresh=0.8, axis_labels=None):
# 利用sklearn中的函数生成混淆矩阵并归一化
    cm = confusion_matrix(y_predicted789, test_y, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例
    # 图像标题
    if title is not None:
        plt.title(title,fontweight='bold')
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=0)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label',fontweight='bold')
    plt.xlabel('Predicted label',fontweight='bold')

    # # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    # for i in range(np.shape(cm)[0]):
    #     for j in range(np.shape(cm)[1]):
    #         if int(cm[i][j] * 100 + 0.5) > 0:
    #             plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
    #                     ha="center", va="center",fontsize=15,
    #                     color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行

    # 仅显示主对角线数值
    for i in range(np.shape(cm)[0]):
        if int(cm[i][i] * 100 + 0.5) > 0:
            plt.text(i, i, format(int(cm[i][i] * 100 + 0.5), 'd') + '%',
                     ha="center", va="center", fontsize=15,
                     color="white" if cm[i][i] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行

    # 显示
    #plt.savefig("四分类混淆矩阵.svg", dpi=300, format="svg", bbox_inches='tight', pad_inches=0)
    plt.show()
if __name__ == '__main__':
    plot_matrix(y_predicted789, test_y,[0, 1, 2, 3], title='confusion_matrix_model',axis_labels=[0,1,2,3])   #y_predprob4 或 y_predict1