
import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

data = pd.read_csv('D:\Desktop\sou_hangbanshuju\meiguo\拆分csv删除缺失值行/new_bili_meiguo2015.csv')
# print(data)
data = data[['ORIGIN_AIRPORT','DESTINATION_AIRPORT','DISTANCE']]
print(data)

data = data.drop_duplicates(subset=['ORIGIN_AIRPORT','DESTINATION_AIRPORT'], keep='first', inplace=False)
print(data)


#将data中包含这40个机场的行数据全部提取出来
airport_name = ['ORD', 'DFW', 'LAX', 'ATL', 'DEN', 'SFO', 'IAH', 'PHX', 'LAS', 'SEA', 'BOS', 'LGA', 'JFK', 'MSP', 'EWR', 'DTW', 'MCO', 'CLT', 'BWI', 'PHL', 'SLC', 'MIA', 'SAN', 'MDW', 'DCA', 'FLL', 'DAL', 'HOU', 'TPA', 'BNA', 'STL', 'PDX', 'OAK', 'AUS', 'CLE', 'SMF', 'SJC', 'SNA', 'MCI', 'IAD']
data = data[data['ORIGIN_AIRPORT'].isin(airport_name) & data['DESTINATION_AIRPORT'].isin(airport_name)]

print(data)

#画距离邻接矩阵 ， #机场的种类，不能重复
rows = ['ORD', 'DFW', 'LAX', 'ATL', 'DEN', 'SFO', 'IAH', 'PHX', 'LAS', 'SEA', 'BOS', 'LGA', 'JFK', 'MSP', 'EWR', 'DTW', 'MCO', 'CLT', 'BWI', 'PHL', 'SLC', 'MIA', 'SAN', 'MDW', 'DCA', 'FLL', 'DAL', 'HOU', 'TPA', 'BNA', 'STL', 'PDX', 'OAK', 'AUS', 'CLE', 'SMF', 'SJC', 'SNA', 'MCI', 'IAD']
cols = ['ORD', 'DFW', 'LAX', 'ATL', 'DEN', 'SFO', 'IAH', 'PHX', 'LAS', 'SEA', 'BOS', 'LGA', 'JFK', 'MSP', 'EWR', 'DTW', 'MCO', 'CLT', 'BWI', 'PHL', 'SLC', 'MIA', 'SAN', 'MDW', 'DCA', 'FLL', 'DAL', 'HOU', 'TPA', 'BNA', 'STL', 'PDX', 'OAK', 'AUS', 'CLE', 'SMF', 'SJC', 'SNA', 'MCI', 'IAD']

Mat = data.pivot('ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DISTANCE').fillna(0).reindex(index=rows, columns=cols, fill_value=0).values

# 打印
print('距离矩阵：')
print(Mat)
print("\n")

new_data = pd.DataFrame(Mat)
juli_xiangguanxing = new_data.corr()

print("距离相关性矩阵：")
print(juli_xiangguanxing)

juli_xiangguanxing.to_csv('meiguo2015_juli40.csv',index=False)


