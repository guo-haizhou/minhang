
import pandas as pd

# 读取csv中Sheet1中的数据
data = pd.DataFrame(pd.read_excel("E:\meiguo2015\将数据集处理成邻接矩阵和特征矩阵\连通性邻接矩阵\美国2015无重复的航线.xlsx", "Sheet1"))

# 查看读取数据内容
print(data)

print(data['ORIGIN_AIRPORT'])

mylist = set(data['ORIGIN_AIRPORT'])
print("出发机场：",mylist)
print("出发机场个数：",len(mylist))

mylist2 = set(data['DESTINATION_AIRPORT'])
print("到达机场：",mylist2)
print("到达机场个数：",len(mylist2))

print("\n")

quanbu_jichang = mylist | mylist2
print("共有",len(quanbu_jichang),"个机场")

quanbu_jichang_liebiao = ','.join(quanbu_jichang)
print(quanbu_jichang_liebiao)

