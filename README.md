# 分类预测
## 1.数据
原始数据网址：https://www.kaggle.com/usdot/flight-delays.  
经过预处理后，数据在github的Release中的tu_data.
## 2.实验
将adasyn+smote+RF与RF、adasyn+RF、smote+RF的性能做对比。针对new_bili_meiguodata2015.csv按照3：1划分为训练集和测试集，训练集训练模型，测试集验证模型分类预测准确率。
