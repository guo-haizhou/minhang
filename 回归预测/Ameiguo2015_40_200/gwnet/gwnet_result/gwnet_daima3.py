import torch, gc
import numpy as np
import argparse
import time
from Ameiguo2015_40_200.gwnet import util6
import matplotlib.pyplot as plt
from engine import *
import shutil
import random
import pickle
from util2 import z_inverse


######################################################################################
#若想运行gwnet代码，将代码中trainx, trainy[:, 0, :, :args.target_length],tpl中的tpl去掉即可

###########################################################################################

# 选取显存最大的显卡
# import pynvml
# pynvml.nvmlInit()
# # 最小占用空间, 显卡编号
# min_used = sys.maxsize
# best_gpu = -1
# for i in range(4):
#     handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#     meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     if meminfo.used < min_used:
#         min_used = meminfo.used
#         best_gpu = i
# 设置显卡的可见性
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
print("使用%d号显卡"%(0))
print(torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
# parser.add_argument('--device',type=str,default='cpu',help='')

# 双向邻接矩阵, 可能考虑到有向图?
# parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
# 单向邻接矩阵, 可能倒履到无向图
parser.add_argument('--adjtype', type=str, default='transition', help='adj type')
# 序列长度, 训练序列长度?
parser.add_argument('--seq_length', type=int, default=15, help='')
parser.add_argument('--target_length', type=int, default=1, help='')
parser.add_argument('--nhid', type=int, default=16, help='')
# 输入维度?
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
# JiNan num_nodes = 561 cluster_nodes = 20
# XiAN num_nodes = 792 cluster_nodes = 40
# PEMS num_nodes = 228 cluster_nodes = None
# parser.add_argument('--num_nodes', type=int, default=228, help='number of nodes')
# 郑州的节点个数 2000
# parser.add_argument('--num_nodes', type=int, default=1968, help='number of nodes')
parser.add_argument('--num_nodes', type=int, default=40, help='number of nodes')
# 批次大小
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
# 学习率
parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate')
# dropout比例
# parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
# 学习率衰减系数
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay rate')
# 迭代次数
parser.add_argument('--epochs', type=int, default=10, help='')
# 每多少个epoch打印一次
parser.add_argument('--print_every', type=int, default=2, help='')
parser.add_argument('--force', type=str, default=False, help="remove params dir", required=False)
# 模型参数保存路径
parser.add_argument('--save', type=str, default='./garage/moxing', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
# gwnet
parser.add_argument('--model',type=str,default='gwnet',help='adj type')
# Gated_STGCN
# parser.add_argument('--model', type=str, default='Gated_STGCN', help='adj type')
# parser.add_argument('--model', type=str, default='ADGCN', help='adj type')
# parser.add_argument('--model', type=str, default='TGCN', help='adj type')
# Attention model
# parser.add_argument('--model', type=str, default='ASTGCN_Recent', help='adj type')
# H_GCN_wh
# parser.add_argument('--model',type=str,default='H_GCN_wh',help='adj type')
parser.add_argument('--decay', type=float, default=0.95, help='decay rate of learning rate ')

args = parser.parse_args()
##model repertition
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# 主训练函数
def main():
    # set seed
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # load data
    device = torch.device(args.device)

    # 加载zhengzhou邻接矩阵(原始矩阵记录的是距离)
    adj_mx = util6.zhengzhou_load_adj("E:\GWNET_daima\Ameiguo2015_40_200\meiguo_data\meiguo2015_hangbanshu40.csv")

    # 郑州数据集
    dataloader, x_stats = util6.zhengzhou_load_dataset(dataset_dir="E:\GWNET_daima\Ameiguo2015_40_200\meiguo_data\重新生成的美国2015延误时间特征矩阵.csv",
                                                      n_route=args.num_nodes,
                                                      batch_size=args.batch_size,
                                                      valid_batch_size=args.batch_size,
                                                      test_batch_size=args.batch_size)
    print(x_stats)

    # 预测可视化样例实验的数据
    ex_dataloader, ex_x_stats = util6.exp_dataloader(dataset_dir="E:\GWNET_daima\Ameiguo2015_40_200\meiguo_data\重新生成的美国2015延误时间特征矩阵.csv")

    # 邻接矩阵放到显存上
    supports = [torch.tensor(adj_mx).cuda()]

    print(args)
    if args.model == 'gwnet':
        engine = trainer1(args.in_dim, args.seq_length,args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay
                          )

    elif args.model == 'ASTGCN_Recent':
        engine = trainer2(args.in_dim, args.seq_length, args.target_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay, x_stats
                          )
    elif args.model == 'GRCN':
        engine = trainer3(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay
                          )
    # 传入训练数据的统计信息
    elif args.model == 'Gated_STGCN':
        print("Gated_STGCN")
        engine = trainer4(args.in_dim, args.seq_length, args.target_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay, x_stats
                          )
    elif args.model == 'H_GCN_wh':
        engine = trainer5(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay
                          )

    elif args.model == 'OGCRNN':
        engine = trainer8(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay
                          )
    elif args.model == 'OTSGGCN':
        engine = trainer9(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                          args.learning_rate, args.weight_decay, device, supports, args.decay
                          )
    elif args.model == 'LSTM':
        engine = trainer10(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay
                           )
    elif args.model == 'GRU':
        engine = trainer11(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay
                           )
    elif args.model == 'TGCN':
        engine = trainer12(args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay, x_stats
                           )

    # attention dynamic graph convolution network
    elif args.model == 'ADGCN':
        engine = trainer13(args.in_dim, args.seq_length, args.target_length, args.num_nodes, args.nhid, args.dropout,
                           args.learning_rate, args.weight_decay, device, supports, args.decay, x_stats
                           )


    # check parameters file
    '''保存网络保存参数'''
    params_path = args.save + "/" + args.model
    if os.path.exists(params_path) and not args.force:
        pass
        # raise SystemExit("Params folder exists! Select a new params path please!")
    else:
        if os.path.exists(params_path):
            shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('Create params directory %s' % (params_path))


    '''开始训练'''
    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []

    '''train loss 和 valid loss'''
    tr_loss_ = list()
    va_loss_ = list()
    te_loss_ = list()

    for i in range(1, args.epochs + 1):
        # if i == 3:
        #     print("epoch 3")
        # if i % 10 == 0:
        # lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
        # for g in engine.optimizer.param_groups:
        # g['lr'] = lr
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        # shuffle
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)

            metrics = engine.train(trainx, trainy[:, 0, :, :args.target_length])
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            if metrics[2] < 0:
               train_mape.append(-metrics[2])
            else:
                train_mape.append(metrics[2])
            train_rmse.append(metrics[3])

        print("trainx的数据类型：",trainx.shape)
        print("trainy的数据类型：",trainy.shape)
            # if iter % args.print_every == 0 :
            #   log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
            #  print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()

        '''验证'''
        dataloader['val_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)

            metrics = engine.eval(testx, testy[:, 0, :, :args.target_length])
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            if metrics[2] < 0:
                valid_mape.append(-metrics[2])
            else:
                valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)       #每次循环后的loss取中位数作为最终的loss
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape,
                         mvalid_rmse, (t2 - t1)), flush=True)
        torch.save(engine.model.state_dict(),
                   params_path + "/" + args.model + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")

        '''集中输出loss'''
        tr_loss_.append(mtrain_loss)
        va_loss_.append(mvalid_loss)


        test_loss = []
        test_mae = []
        test_rmse = []
        test_mape = []
        dataloader['test_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            with torch.no_grad():
                metrics = engine.eval(testx, testy[:, 0, :, :args.target_length])
                test_loss.append(metrics[0])
                test_mae.append(metrics[1])
                if metrics[2] < 0:
                    test_mape.append(-metrics[2])
                else:
                    test_mape.append(metrics[2])
                test_rmse.append(metrics[3])

        te_loss_.append(np.mean(test_loss))

        #######################################
        #跑完一个poch后，清理内存
        gc.collect()
        torch.cuda.empty_cache()
        ######################################

    #######################################################################################
    # 将训练集loss存储在csv中，方便画损失图
    import pandas as pd
    cun_train_loss = pd.DataFrame({'train_loss': tr_loss_})
    # cun_train_loss.to_csv('E:\GWNET_daima\DCRNN/flight_data/flight_train_loss.csv', index=False)

    # 将测试集loss存储在csv中，方便画损失图
    import pandas as pd
    cun_test_loss = pd.DataFrame({'test_loss': te_loss_})
    # cun_test_loss.to_csv('E:\GWNET_daima\DCRNN/flight_data/flight_test_loss.csv', index=False)
    ##########################################################################################


    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    print("train loss", tr_loss_)
    print("valid loss", va_loss_)

    # 最佳模型
    bestid = np.argmin(his_loss)

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    '''test'''
    # 加载模型
    engine.model.load_state_dict(torch.load(
        params_path + "/" + args.model + "_epoch_" + str(bestid + 1) + "_" + str(round(his_loss[bestid], 2)) + ".pth"))
    engine.model.eval()

    test_loss = []
    test_mae = []
    test_mape = []
    test_rmse = []

    '''case study实验'''

    torch.cuda.empty_cache()  # 释放GPU显存
    print(torch.cuda.memory_summary())  # 查看显存信息

    for iter, (x, y) in enumerate(ex_dataloader['ex_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)


        with torch.no_grad():
            mae, rmse, predicts = engine.ex_eval(testx, testy[:, 0, :, :args.target_length])
            print(predicts.shape)    #([72, 1, 716, 12])
            print(testx.shape)       #([72, 1, 716, 12])
            print(testy.shape)       #([72, 1, 716, 1])
            ''' 保存真实值和预测结果'''
            '''预测值需要反归一化'''



            predicts = z_inverse(predicts, x_stats['mean'], x_stats['std'])
            testx = z_inverse(testx, x_stats['mean'], x_stats['std'])



            print('预测值：',predicts[1][0][7])
            print('真实值：', testy[0][0][7])


            #######################################################################################################################
            #将预测值和真实值保存成npy文件，都是三维的，即预测值是200*325*10，需处理为200*325*1，真实值为200*325*1数据，方便计算误差
            baocun_predicts = predicts.squeeze().cpu().numpy()
            np.save(file="E:\GWNET_daima\Ameiguo2015_40_200\gwnet\gwnet_result/meiguo2015_gwnet_hangbanshu_predicts200.npy", arr=baocun_predicts)

            baocun_testy = testy.squeeze().cpu().numpy()
            #np.save(file="E:\GWNET_daima\Ameiguo2015_40_200\gwnet\gwnet_result/meiguo2015_gwnet_testy200.npy", arr=baocun_testy)

            ########################################################################################################################

            #################################################################################################
            # # 第一种：以预测的12个值的平均值为最终的预测值
            # true = []
            # pred = []
            # # 测试集test只有72行数据
            # for j in range(72):
            #     # print(predicts[0][0][0])  #第一个0代表预测的72行数中第1个数，第二个0不用变，第三个0代表预测的第0列(即716列中的第1列)
            #     k = predicts[j][0][1].squeeze().cpu().numpy()
            #     k = list(k)
            #     pred.append(np.mean(k))
            #     v = testy[j][0][1].squeeze().cpu().numpy()
            #     true.append(v)
            #     print("预测出12个值的平均值：", np.mean(k), "真实值：", v)
            #     # print(testx[0][0][0])
            #     # print(testy[0][0][0])
            ################################################################################################

            #################################################################################################
            # 第二种：以预测的12个值中最接近真实值的那个值为最终的预测值
            xin_predict = []   #将预测的12个值中最接近真实值的值写入这个列表中。

            # true = []
            # pred = []
            # # 测试集test只有72行数据
            # for j in range(250):
            #     # print(predicts[0][0][0])  #第一个0代表预测的72行数中第1个数，第二个0不用变，第三个0代表预测的第0列(即716列中的第1列)
            #     k = predicts[j][0][0].squeeze().cpu().numpy()
            #     k = list(k)
            #
            #     v = testy[j][0][0].squeeze().cpu().numpy()
            #     true.append(v)
            #
            #     zuijiejin_pred = min(k, key=lambda x: abs(x - v))
            #     pred.append(zuijiejin_pred)
            #     print("预测出12个值中最接近真实值的那个值：", zuijiejin_pred, "真实值：", v)
            #     # print(testx[0][0][0])
            #     # print(testy[0][0][0])
            #
            #
            #     #numpy数据转tensor数据
            #     b = np.array(zuijiejin_pred)
            #     predicts[j][0][0] = torch.from_numpy(b)
            #
            #
            #
            # print("新的预测值的shape：",predicts.shape)
            # print("真实值的shape：",testy.shape)
            #
            # #####################################################################################################
            #
            # true = [true[i].item() for i in range(len(true))]
            # print("第 列的全部真实值：", true)
            # print("第 列的全部预测值：", pred)
            # print(len(true))
            # print(len(pred))
            #
            #
            # #将真实值和预测值保存起来
            # import pandas as pd
            # data = pd.DataFrame({'true': true, 'pred': pred})
            # data.to_csv(args.model + "训练结果" + ".csv", index=False)


            # np.savetxt(X=predicts.squeeze().cpu().numpy().reshape(-1,1), fname=args.model + "预测结果" + ".txt")
            # np.savetxt(X=testy.squeeze().cpu().numpy().reshape(-1,1), fname=args.model + "真实值" + ".txt")

    dataloader['test_loader'].shuffle()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)
        with torch.no_grad():
            metrics = engine.eval(testx, testy[:, 0, :, :args.target_length])
            test_loss.append(metrics[0])
            test_mae.append(metrics[1])
            if metrics[2] < 0:
                test_mape.append(-metrics[2])
            else:
                test_mape.append(metrics[2])
            test_rmse.append(metrics[3])

    mtest_loss = np.mean(test_loss)
    mtest_mae = np.mean(test_mae)
    mtest_mape = np.mean(test_mape)
    mtest_rmse = np.mean(test_rmse)

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
    print(log.format(mtest_mae, mtest_rmse, mtest_mape))

    return mtest_mae, mtest_rmse, mtest_mape

'''加载邻接表'''
def load_adj_list(file_path):
    adj_list = list()
    with open(file_path, "rb") as fr:
        adj_list = pickle.load(fr)
    return adj_list


if __name__ == "__main__":

    # t1 = time.time()
    # main()
    # # 是同ADGCN
    # # main_transiton()
    # t2 = time.time()
    # print("Total time spent: {:.4f}".format(t2 - t1))


    # 重复跑若干次
    mae_list = list()
    rmse_list = list()
    mape_list = list()



    for i in range(1):
        mae, rmse, mape = main()
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)

    print("avg_MAE = %.2f, avg_RMSE = %.2f, avg_MAPE = %.2f" % (sum(mae_list)/len(mae_list), sum(rmse_list)/len(rmse_list), sum(mape_list)/len(mape_list)))