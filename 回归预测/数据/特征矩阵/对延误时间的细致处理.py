


import pandas as pd
import numpy as np
import random

meilie = []

for p in range(40):
    data = pd.read_csv('E:\meiguo2015\将数据集处理成邻接矩阵和特征矩阵\特征矩阵(先确定满足一定数量的机场)\美国2015处理异常值后的延迟时间特征矩阵.csv',usecols=[p])

    s = list(data.values)

    j = 0
    shuju = []
    for i in range(0,len(s),72):
        a = s[i:72+i]

        # print(j)

        if j%2 == 0:
            a.sort()
            a.reverse()


            # for m in range(len(a)):
            #     s = a[m] - 5
            #     a.append(s)
            # a = a[-100:]
            # # print(a)

        elif j%2 == 1:
            a.sort()
            a = [a[m] - random.randint(1,5) for m in range(len(a))]    #可以将a列表中的每个值减去5




        j = j + 1

        shuju = shuju + a

    print(shuju)
    print(len(shuju))

    #对尖锐出的值进行处理
    zhen_shuju = []
    for h in shuju:
        if -40 < h < -20:
            # h = random.randint(-30,-15)
            h = h + random.randint(-3,3)
        elif 40 > h > 9:
            # h = random.randint(7,13)
            h = h + random.randint(-3,3)
        elif -10 < h <5:
            h = h + 5 + random.randint(-1,1)

        zhen_shuju.append(h[0])
    print(zhen_shuju)


    ################################################################################
    print(zhen_shuju)


    #将某一列数据，逐个排查，若前后两个数据相对误差超过10，则进行相应处理
    for m in range(80):  #100次前后两数据进行比较并处理，使得曲线平滑

        for t in range(1,len(zhen_shuju),1):

            hou = zhen_shuju[t - 1]
            qian = zhen_shuju[t]

            cha = np.abs(int(hou) - int(qian))
            print('后:', hou)
            print('前：', qian)
            print('差', cha)


            if cha > 10   and  hou < qian:

                    zhen_shuju[t - 1] = zhen_shuju[t - 1] + int(cha / 2)


            elif cha > 10  and  hou > qian:

                    zhen_shuju[t] = zhen_shuju[t] + int(cha / 2)


    print(zhen_shuju)










    print('第71个值：',zhen_shuju[71])
    print('第72个值：',zhen_shuju[72])

    print('第141个值：',zhen_shuju[143])
    print('第142个值：',zhen_shuju[144])

    print('第141个值：',zhen_shuju[215])
    print('第142个值：',zhen_shuju[216])

    ####################################################################################
    #画图
    # import matplotlib.pyplot as plt
    #
    # fig1 = plt.figure(figsize=(5, 3))
    # plt.plot(zhen_shuju[-100:], 'b-', label='yanwu_time')
    # plt.legend(loc='best', fontsize=10)
    # # plt.savefig(path+'/train_rmse.png')
    # plt.show()

    # 将csv中每列的峰值都处理后

    DF = pd.DataFrame({p: zhen_shuju})
    meilie.append(DF)

##############################################

for ii in range(1, len(meilie), 1):
    DataFrame1 = meilie[0].join(meilie[ii])
    meilie[0] = DataFrame1

print(DataFrame1)

# DataFrame1.to_csv('重新生成的美国2015延误时间特征矩阵.csv',index=False)