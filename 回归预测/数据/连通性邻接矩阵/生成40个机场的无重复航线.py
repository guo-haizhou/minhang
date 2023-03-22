
import pandas as pd

data = pd.read_excel('E:\meiguo2015\将数据集处理成邻接矩阵和特征矩阵\连通性邻接矩阵\美国2015无重复的航线.xlsx')
print(data)

print("超过2000数量的机场有：")
print(['ORD', 'DFW', 'LAX', 'ATL', 'DEN', 'SFO', 'IAH', 'PHX', 'LAS', 'SEA', 'BOS', 'LGA', 'JFK', 'MSP', 'EWR', 'DTW', 'MCO', 'CLT', 'BWI', 'PHL', 'SLC', 'MIA', 'SAN', 'MDW', 'DCA', 'FLL', 'DAL', 'HOU', 'TPA', 'BNA', 'STL', 'PDX', 'OAK', 'AUS', 'CLE', 'SMF', 'SJC', 'SNA', 'MCI', 'IAD'])
s = ['ORD', 'DFW', 'LAX', 'ATL', 'DEN', 'SFO', 'IAH', 'PHX', 'LAS', 'SEA', 'BOS', 'LGA', 'JFK', 'MSP', 'EWR', 'DTW', 'MCO', 'CLT', 'BWI', 'PHL', 'SLC', 'MIA', 'SAN', 'MDW', 'DCA', 'FLL', 'DAL', 'HOU', 'TPA', 'BNA', 'STL', 'PDX', 'OAK', 'AUS', 'CLE', 'SMF', 'SJC', 'SNA', 'MCI', 'IAD']
print("共",len(s),"个")

#出发机场列含有这40个元素的行，则保存对应行的数据
print(data[data['ORIGIN_AIRPORT'].isin(['ORD', 'DFW', 'LAX', 'ATL', 'DEN', 'SFO', 'IAH', 'PHX', 'LAS', 'SEA', 'BOS', 'LGA', 'JFK', 'MSP', 'EWR', 'DTW', 'MCO', 'CLT', 'BWI', 'PHL', 'SLC', 'MIA', 'SAN', 'MDW', 'DCA', 'FLL', 'DAL', 'HOU', 'TPA', 'BNA', 'STL', 'PDX', 'OAK', 'AUS', 'CLE', 'SMF', 'SJC', 'SNA', 'MCI', 'IAD'])])
###################################################

data = data[data['ORIGIN_AIRPORT'].isin(['ORD', 'DFW', 'LAX', 'ATL', 'DEN', 'SFO', 'IAH', 'PHX', 'LAS', 'SEA', 'BOS', 'LGA', 'JFK', 'MSP', 'EWR', 'DTW', 'MCO', 'CLT', 'BWI', 'PHL', 'SLC', 'MIA', 'SAN', 'MDW', 'DCA', 'FLL', 'DAL', 'HOU', 'TPA', 'BNA', 'STL', 'PDX', 'OAK', 'AUS', 'CLE', 'SMF', 'SJC', 'SNA', 'MCI', 'IAD'])]
print(data[data['DESTINATION_AIRPORT'].isin(['ORD', 'DFW', 'LAX', 'ATL', 'DEN', 'SFO', 'IAH', 'PHX', 'LAS', 'SEA', 'BOS', 'LGA', 'JFK', 'MSP', 'EWR', 'DTW', 'MCO', 'CLT', 'BWI', 'PHL', 'SLC', 'MIA', 'SAN', 'MDW', 'DCA', 'FLL', 'DAL', 'HOU', 'TPA', 'BNA', 'STL', 'PDX', 'OAK', 'AUS', 'CLE', 'SMF', 'SJC', 'SNA', 'MCI', 'IAD'])])

#保存这40个机场之间的无重复航线数据
new_hangxian = data[data['DESTINATION_AIRPORT'].isin(['ORD', 'DFW', 'LAX', 'ATL', 'DEN', 'SFO', 'IAH', 'PHX', 'LAS', 'SEA', 'BOS', 'LGA', 'JFK', 'MSP', 'EWR', 'DTW', 'MCO', 'CLT', 'BWI', 'PHL', 'SLC', 'MIA', 'SAN', 'MDW', 'DCA', 'FLL', 'DAL', 'HOU', 'TPA', 'BNA', 'STL', 'PDX', 'OAK', 'AUS', 'CLE', 'SMF', 'SJC', 'SNA', 'MCI', 'IAD'])]
new_hangxian.to_excel("美国2015无重复航线40.xlsx",index=False)