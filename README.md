# 分类预测
## 1.数据
原始数据网址：https://www.kaggle.com/usdot/flight-delays.  
经过预处理后，数据在github的Release中的tu_data.
## 2.实验
将adasyn+smote+RF与RF、adasyn+RF、smote+RF的性能做对比。针对new_bili_meiguo2015.csv按照3：1划分为训练集和测试集，训练集训练模型，测试集验证模型分类预测准确率。
  
![image](https://user-images.githubusercontent.com/75230726/227077317-58853392-1987-4f9e-bd4c-7251e07d7b39.png)  
用ADASYN算法和SMOTE算法平衡数据中的分类类别数量。  
根据qingkuang/adasyn+smote+RF.py得到混合平衡采样后的模型性能，并比较未平衡采样的模型、用adasyn平衡采样后的模型性能以及smote平衡采样后的模型性能，用宏平均.py和权重平均.py画图如下：  
  
![image](https://user-images.githubusercontent.com/75230726/226902847-212acf49-3348-414e-b178-026cb107d07d.png)          ![image](https://user-images.githubusercontent.com/75230726/226902896-00150145-02d3-498d-874f-690cbb35ecb5.png)   
通过比较四种情况下模型性能的宏平均和权重平均的性能，发现融合ADASYN算法和SMOTE算法生成的训练集情况4的训练效果最优。  
   
根据qingkuang/adasyn+smote+RF.py的性能与对照文献“基于非线性赋权XGBoost算法的航班延误分类预测”的性能做比较，模型1为朴素贝叶斯Naive Bayesian算法，模型2为支持向量机SVM算法，模型3为K值近邻KNN算法，模型4为随机森林RF算法，模型5为ADASYN+SMOTE+RF算法，用性能柱状图.py画图如下：  
  
![image](https://user-images.githubusercontent.com/75230726/226903310-ef8efd1a-f69e-4dd0-92a9-402d7d5eee71.png)  






