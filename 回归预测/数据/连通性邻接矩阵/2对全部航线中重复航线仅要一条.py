import pandas as pd

# 读取csv中Sheet1中的数据
data = pd.DataFrame(pd.read_excel("E:\meiguo2015\将数据集处理成邻接矩阵和特征矩阵\连通性邻接矩阵\美国2015全部航线.xlsx", "Sheet1"))

# 查看读取数据内容
print(data)

# 查看是否有重复行
re_row = data.duplicated()
print(re_row)

# 查看去除重复行的数据
no_re_row = data.drop_duplicates()
print(no_re_row)

no_re_row.to_excel("美国2015无重复的航线.xlsx",index=False)

#######################################################################################################
#将航线打印成生成邻接矩阵的格式
for j in range(len(no_re_row.values)):
    b = no_re_row.values[j][0]+','+no_re_row.values[j][1]
    # print(b)

    # addEdge(self._adjlist, LLF, HGH)
    print('addEdge(M,'+b+')')


##########################################################################################