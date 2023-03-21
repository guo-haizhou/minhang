

import pandas as pd

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

data = pd.read_csv('D:\Desktop\sou_hangbanshuju\meiguo\拆分csv删除缺失值行/new_bili_meiguo2015.csv')
print(data)

airport_name = ['ORD', 'DFW', 'LAX', 'ATL', 'DEN', 'SFO', 'IAH', 'PHX', 'LAS', 'SEA', 'BOS', 'LGA', 'JFK', 'MSP', 'EWR', 'DTW', 'MCO', 'CLT', 'BWI', 'PHL', 'SLC', 'MIA', 'SAN', 'MDW', 'DCA', 'FLL', 'DAL', 'HOU', 'TPA', 'BNA', 'STL', 'PDX', 'OAK', 'AUS', 'CLE', 'SMF', 'SJC', 'SNA', 'MCI', 'IAD']


# #
# yanwushijian_ORD = data[data['ORIGIN_AIRPORT'] == 'ORD']['ARRIVAL_DELAY'][:2000].values
# yanwushijian_ORD = pd.DataFrame(yanwushijian_ORD,columns=['ORD'])
# print(yanwushijian_ORD)
#
# #
# yanwushijian_DFW = data[data['ORIGIN_AIRPORT'] == 'DFW']['ARRIVAL_DELAY'][:2000].values
# yanwushijian_DFW = pd.DataFrame(yanwushijian_DFW,columns=['DFW'])
# print(yanwushijian_DFW)
#
# #


yanwu = []
for i in range(40):
    yanwushijian = data[data['ORIGIN_AIRPORT'] == airport_name[i]]['ARRIVAL_DELAY'][:2000].values
    yanwushijian = pd.DataFrame(yanwushijian, columns=[airport_name[i]])
    print(yanwushijian)
    yanwu.append(yanwushijian)

print(yanwu)


s = pd.concat(yanwu, axis=1)
print(s)

s.to_csv('meiguo2015_yanwu_time(有大量异常值).csv',index=False)


