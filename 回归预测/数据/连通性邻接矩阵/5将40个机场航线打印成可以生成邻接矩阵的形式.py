

import pandas as pd
data = pd.read_excel('E:\meiguo2015\将数据集处理成邻接矩阵和特征矩阵\连通性邻接矩阵\美国2015无重复航线40.xlsx')

#将航线打印成生成邻接矩阵的格式
for j in range(len(data.values)):
    b = data.values[j][0]+','+data.values[j][1]
    # print(b)

    # addEdge(self._adjlist, LLF, HGH)
    print('addEdge(M,'+b+')')
