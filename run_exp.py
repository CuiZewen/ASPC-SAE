import warnings

warnings.filterwarnings('ignore')  ##抑制第三方警告 www.voidcn.com/article/p-eocwwgtb-btd.html
from ASPC import ASPC
import os
import csv
from time import time
from keras.optimizers import Nadam, SGD   #这里进行了改动adam换成nadam 启发网站 https://blog.csdn.net/u012759136/article/details/52302426/
import numpy as np
from keras import backend as K   ##这是学长给的建议，我也不懂，照猫画虎吧，到时候问问##https://zhuanlan.zhihu.com/p/84831334
K.set_image_data_format('channels_last')    ##可以参考这篇文章 https://zhuanlan.zhihu.com/p/89558763
from datasets import load_data

#use_multiprocessing=True  这个地方我一直在考虑，因为最开始用的mac跑出来之后，换成win10一直运行不出来，之后把线程全删掉才能跑出了，前前后后还pip安装了好几个东西，
##对了，还有一点，我之前用的pip是清华的源，清华的很多东西都没有
##pip install pydotplus pip install graphviz 这几个要换成官网的源才能跑，我是这样的

def run_exp(dbs, da_s1, da_s2, expdir, ae_weights_dir, trials=5, verbose=0,
            pretrain_epochs=50, finetune_epochs=50, use_multiprocessing=True):
    '''

    :param dbs: 输入数据
    :param da_s1: 是否使用数据迭代器,即分批次训练
    :param da_s2: 是否数据增强
    :param expdir: 结果输出路径
    :param ae_weights_dir: 预训练模型的权重路径
    :param trials: 训练模型的个数
    :param verbose: 是否在标准输出流输出日志信息，为0不输出，为一输出相关进度条
    :param pretrain_epochs: 预训练的批次
    :param finetune_epochs: 微调模型的训练批次
    :param use_multiprocessing: 多进程处理
    :return:
    '''
    # Log files 判断是否存在结果文件夹
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    logfile = open(expdir + '/results.csv', 'a')  # 保存预测的相关指标
    logwriter = csv.DictWriter(logfile, fieldnames=['trials', 'acc', 'nmi', 'time'])
    logwriter.writeheader()  # 写入表头，即标题

    # Begin training on different datasets
    for db in dbs:  # 对多个数据集进行迭代训练
        logwriter.writerow(dict(trials=db, acc='', nmi='', time=''))

        # load dataset
        x, y = load_data(db)

        # setting parameters
        n_clusters = len(np.unique(y))  # 获取类别数
        dims = [x.shape[-1], 500, 500, 2000, 10]
        # Training
        results = np.zeros(shape=[trials, 3], dtype=float)  # init metrics before finetuning
        for i in range(trials):  # base
            t0 = time()
            save_dir = os.path.join(expdir, db, 'trial%d' % i)#根据轮次创建对应路径
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # prepare model
            model = ASPC(dims, n_clusters)  # 初始化模
            model.compile(optimizer=Nadam(1e-6), loss='mse')  # 模型编译，这里因为因为换成了NADAM，所以记得改一下
                    ##https://www.csdn.net/gather_23/MtTaggxsODI3OC1ibG9n.html  TODO 1e-6的考虑这个地方纯粹感觉比0.0001小，应该可以再优化吧
            # pretraining
            ae_weights = 'ae_weights.h5'
            if ae_weights_dir is None:  # 加载预训练参数  这里因为Nadam比adam和SGD都好，所以都换成Nadam
                model.pretrain(x, y,optimizer=Nadam(1e-3), epochs=pretrain_epochs,
                               save_dir=save_dir, da_s1=da_s1, verbose=verbose)
                ae_weights = os.path.join(save_dir, ae_weights)
            else:
                ae_weights = os.path.join(ae_weights_dir, db, 'trial%d' % i, ae_weights)

            # finetuning
            results[i, :2] = model.fit(x, y, epochs=finetune_epochs if db != 'fmnist' else 10,
                                       da_s2=da_s2, save_dir=save_dir, ae_weights=ae_weights)
  #                                     use_multiprocessing=use_multiprocessing)  这里的话，之前用linux mac是加的，如果win需要删掉，就类似这样

            results[i, 2] = time() - t0

        for t, line in enumerate(results):
            logwriter.writerow(dict(trials=t, acc=line[0], nmi=line[1], time=line[2]))
        mean = np.mean(results, 0)
        logwriter.writerow(dict(trials='avg', acc=mean[0], nmi=mean[1], time=mean[2]))
        logfile.flush()

    logfile.close()  # 关闭文件读写

#查看编码器的隠层输出（比葫芦画瓢，哈哈哈哈）  导入数据，导出数据，这样不需要预训练
def load_model(db,i=0,expdir='result/'):    ##导入
    '''

    :param db: 需要查看的数据集对应的模型
    :param i: 第几次实验的结果
    :param expdir: 保存模型的位置
    :return:
    '''
    # Log files 判断是否存在结果文件夹
    if not os.path.exists(expdir):
        raise("你应该先训练一下模型")
    else:
        if db in ['mnist', 'mnist-test', 'usps','fmnist']:
            # load dataset
            x, y = load_data(db)
            # setting parameters
            n_clusters = len(np.unique(y))  # 获取类别数  去除重复的数字，获取类别
            dims = [x.shape[-1], 500, 500, 2000, 10]   ##这和之前的保持一致
            load_path = expdir+db+'/trial%d' % i +'/model_best.h5'  ##把训练的导入，可以说暂存，这个我纯粹照搬网上的，我也不懂
            from MyModel import autoencoder
            AE,Encoder = autoencoder(dims)
            Encoder.load_weights(load_path)
            y_pred = Encoder.predict(x)
            #print("输出稀疏编码器前五个样本最后的隠层输出",y_pred[0:5,:])
            #print("Using k-means for initialization by default.")
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
            #print("样本总数",x.shape)
            cluster_pred = kmeans.fit_predict(X=y_pred)
            centers = kmeans.cluster_centers_.astype(np.float32)
            #print("初步聚类的前五个样本对应的预测值",cluster_pred[0:5])
            #print('由于聚类标签划分标准未知，所以聚类标签与实际标签可能并不相符，故需要对齐,-1表示噪声')
            cluster_pred,reflect = equal_cluster_label(y,cluster_pred)
            #print("输出预测映射值", reflect)
            #print("映射关系，实际标签0～9对应的聚类预测值：",cluster_pred[0:5])
            from sklearn.metrics import accuracy_score
            print("输出前五个样本对应的实际值",y[0:5])
            #print("预测的准确率:",accuracy_score(y,cluster_pred))
        else:
            raise("你应该将新的数据集添加到表单中")

##映射标签   https://blog.csdn.net/yiran103/article/details/103826367?depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-8&utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-8
def equal_cluster_label(y_true,y_pred):
    '''

    :param y_true: 实际标签
    :param y_pred: 预测值
    :return:
     reflect:关系映射
     pred_label:转换后的预测值
    '''
    Label = np.unique(y_true)  # 去除重复的元素，由小大大排列
    nClass = len(Label)  # 标签的大小
    G = np.zeros((nClass, nClass))  ##返回n，n的0矩阵
    for i in range(nClass):
        ind_cla1 = y_true == Label[i]
        ind_cla1 = ind_cla1.astype(float)  ##astype用来转换数据类型  TODO：这个地方不懂：此处用float而不是np.float64,Numpy会将python类型映射到等价的dtype上
        for j in range(nClass):
            ind_cla2 = y_pred == Label[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)  ##加起来
    import munkres   ##这里考虑用匈牙利算法，网上是这么叫的，就是映射标签
    m = munkres.Munkres()
    index = m.compute(-G.T)   ##转置一下
    index = np.array(index)
    #映射关系
    # print(index)
    c = index[:, 1]  ##获取第一个数据
    reflect = c
    pred_label = []
    for value in y_pred:
        if value ==-1:
            pred_label.append(-1)
        else:
            pred_label.append(reflect[value])
    return pred_label,reflect


if __name__ == "__main__":
    # Global experiment settings
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用GPU训练，但是这里遇到一个问题，是否需要这样做呢，用在普通的python代码上，https://blog.csdn.net/qq_42815385/article/details/88582035
    expdir = 'result'  # 结果文件
    ae_weight_root = None  # 'result'
    trials = 5 #500
    verbose = 1
    dbs = ['mnist', 'mnist-test', 'usps','fmnist']  #数据集,默认全部数据集都会被迭代训练
    pretrain_epochs = 200 #500
    finetune_epochs = 100 #100
 #   use_multiprocessing = False  # if encounter errors, set it to False 多线程处理，不支持win10

#    run_exp(dbs, da_s1=True, da_s2=True,
 #           pretrain_epochs=pretrain_epochs,
  #          finetune_epochs=finetune_epochs,
 # #          use_multiprocessing=use_multiprocessing,
    #        expdir=expdir,
     #       ae_weights_dir=ae_weight_root,
      #      verbose=verbose, trials=trials)
    load_model('fmnist', i=3, expdir='result/')
