import os
from time import strftime, localtime
import torch
import numpy as np
import torch.nn.functional as F
from utils import get_msg_mgr, mkdir, MeanIOU


def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # n c p
        y = F.normalize(y, p=2, dim=1)  # n c p
    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist/num_bin if metric == 'cos' else dist / num_bin

# Exclude identical-view cases


def de_diag(acc, each_angle=False):
    dividend = acc.shape[1] - 1.
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result

# Modified From https://github.com/AbnerHqC/GaitSet/blob/master/model/utils/evaluator.py


def identification(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    # sample_num = len(feature)

    probe_seq_dict = {'CASIA-B': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']]}

    gallery_seq_dict = {'CASIA-B': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}
    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]),
                    view_num, view_num, num_rank]) - 1.
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                        view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                        view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x, metric)
                    idx = dist.sort(1)[1].cpu().numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)
    result_dict = {}
    np.set_printoptions(precision=3, suppress=True)
    if 'OUMVLP' not in dataset:
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d (Include identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                np.mean(acc[0, :, :, i]),
                np.mean(acc[1, :, :, i]),
                np.mean(acc[2, :, :, i])))
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d (Exclude identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                de_diag(acc[0, :, :, i]),
                de_diag(acc[1, :, :, i]),
                de_diag(acc[2, :, :, i])))
        result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, i])
        result_dict["scalar/test_accuracy/BG"] = de_diag(acc[1, :, :, i])
        result_dict["scalar/test_accuracy/CL"] = de_diag(acc[2, :, :, i])
        np.set_printoptions(precision=2, floatmode='fixed')
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: {}'.format(de_diag(acc[0, :, :, i], True)))
            msg_mgr.log_info('BG: {}'.format(de_diag(acc[1, :, :, i], True)))
            msg_mgr.log_info('CL: {}'.format(de_diag(acc[2, :, :, i], True)))
    else:
        msg_mgr.log_info('===Rank-1 (Include identical-view cases)===')
        msg_mgr.log_info('NM: %.3f ' % (np.mean(acc[0, :, :, 0])))
        msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
        msg_mgr.log_info('NM: %.3f ' % (de_diag(acc[0, :, :, 0])))
        msg_mgr.log_info(
            '===Rank-1 of each angle (Exclude identical-view cases)===')
        msg_mgr.log_info('NM: {}'.format(de_diag(acc[0, :, :, 0], True)))
        result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, 0])



    # # T-SNE
    import torch
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import joblib


    # 对样本进行预处理并画图
    def plot_embedding(data, label, seq_type, title):
        """
        :param data:数据集
        :param label:样本标签
        :param title:图像标题
        :return:图像
        """
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)     # 对数据进行归一化处理
        fig = plt.figure()      # 创建图形实例
        # ax = plt.subplot(111)       # 创建子图
        # 遍历所有样本
        for i in range(data.shape[0]):
            # 在图中为每个数据点画出标签
            # plt.text(data[i, 0], data[i, 1], str(label[i][1:] + '-' + seq_type[i][0:2]), color=plt.cm.Set1(i // 11),
            #             fontdict={'weight': 'bold', 'size': 7})
            plt.text(data[i, 0], data[i, 1], '.', color=plt.cm.Set1(i // 11),
                        fontdict={'weight': 'bold', 'size': 18})
        plt.xticks()        # 指定坐标的刻度
        plt.yticks()
        plt.axis('off')
        plt.title(title, fontsize=14)
        # 返回值
        return fig


    feature_tensor = torch.tensor(feature)
    feature_numpy = feature_tensor.view(feature_tensor.size(0), -1)[0:880:10, :].data.cpu().numpy()
    X_tsne = TSNE(n_components=2,init="pca").fit_transform(feature_numpy)
    fig = plot_embedding(X_tsne, label, seq_type, 't-SNE Embedding of CASIA-B Test')
    

    plt.savefig('/home/huangpanjian/Final_GaitApp/GaitApp_CASIAB/GaitApp_GaitSet/opengait/tsne_test.jpg')
    print("finish")


    return result_dict



def identification_all(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()

    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    # print(label.len())
    # print(seq_type.len())
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    # sample_num = len(feature)

    probe_seq_dict = {'CASIA-B': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                     'OUMVLP': [['00']],
                     'FVG_CCVID_WS': [['1_04', '1_05', '1_06', '1_07', '1_08', '1_09', '2_04', '2_05', '2_06']],
                     'FVG_CCVID_BGHT':[['1_10', '1_11', '1_12']],
                     'FVG_CCVID_CL':[['2_07', '2_08', '2_09']],
                     'FVG_CCVID_MP':[['2_10', '2_11', '2_12']],
                     'FVG_CCVID_ALL':[['1_01', '1_03', '1_04', '1_05', '1_06', '1_07', '1_08', '1_09', '1_10', '1_11', '1_12',
                                       '2_01', '2_03', '2_04', '2_05', '2_06', '2_07', '2_08', '2_09', '2_10', '2_11', '2_12',
                                       '3_01', '3_02', '3_03', '3_04', '3_05', '3_06', '3_07', '3_08', '3_09', '3_10', '3_11', '3_12']],
                     'CCVID_CC':[['1_01', '1_02', '1_03',
                               '2_01', '2_02', '2_03', '2_04', '2_05', '2_06', '2_10', '2_11', '2_12',
                               '3_01', '3_02', '3_03', '3_04', '3_05', '3_06', '3_07', '3_08', '3_09', '3_10', '3_11', '3_12']],
                     'OutdoorGait': [['L_scene1_nm_1', 'L_scene1_nm_2', 'L_scene1_nm_3', 'L_scene1_nm_4', 'L_scene2_nm_1', 'L_scene2_nm_2', 'L_scene2_nm_3', 'L_scene2_nm_4'], 
                                  ['L_scene1_bg_1', 'L_scene1_bg_2', 'L_scene1_bg_3', 'L_scene1_bg_4', 'L_scene2_bg_1', 'L_scene2_bg_2', 'L_scene2_bg_3', 'L_scene2_bg_4'],
                                  ['L_scene1_cl_1', 'L_scene1_cl_2', 'L_scene1_cl_3', 'L_scene1_cl_4', 'L_scene2_cl_1', 'L_scene2_cl_2', 'L_scene2_cl_3', 'L_scene2_cl_4']],
                     }
    gallery_seq_dict = {'CASIA-B': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']],
                        'FVG_CCVID_WS':[['1_02', '2_02']],
                        'FVG_CCVID_BGHT':[['1_02']],
                        'FVG_CCVID_CL':[['2_02']],
                        'FVG_CCVID_MP':[['2_02']],
                        'FVG_CCVID_ALL':[['1_02', '2_02']],
                        'CCVID_CC':[['1_04', '1_05', '1_06', '1_07', '1_08', '1_09', '1_10', '1_11', '1_12',
                                  '2_07', '2_08', '2_09',
                                  '1_01_3', '1_02_3', '1_03_3', '1_04_3', '1_05_3', '1_06_3', '1_07_3', '1_08_3', '1_09_3', '1_10_3', '1_11_3', '1_12_3']],
                        'OutdoorGait': [['L_scene3_nm_1', 'L_scene3_nm_2', 'L_scene3_nm_3', 'L_scene3_nm_4']],
                        }
    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]),
                    view_num, view_num, num_rank]) - 1.
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                        view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                        view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x, metric)
                    idx = dist.sort(1)[1].cpu().numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)
    result_dict = {}
    if 'CASIA-B' in dataset:
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d (Include identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                np.mean(acc[0, :, :, i]),
                np.mean(acc[1, :, :, i]),
                np.mean(acc[2, :, :, i])))
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d (Exclude identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                de_diag(acc[0, :, :, i]),
                de_diag(acc[1, :, :, i]),
                de_diag(acc[2, :, :, i])))
        result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, i])
        result_dict["scalar/test_accuracy/BG"] = de_diag(acc[1, :, :, i])
        result_dict["scalar/test_accuracy/CL"] = de_diag(acc[2, :, :, i])
        np.set_printoptions(precision=2, floatmode='fixed')
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: {}'.format(de_diag(acc[0, :, :, i], True)))
            msg_mgr.log_info('BG: {}'.format(de_diag(acc[1, :, :, i], True)))
            msg_mgr.log_info('CL: {}'.format(de_diag(acc[2, :, :, i], True)))
    elif 'OutdoorGait'in dataset:
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d (Include identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                np.mean(acc[0, :, :, i]),
                np.mean(acc[1, :, :, i]),
                np.mean(acc[2, :, :, i])))
    elif 'OUMVLP' in dataset:
        msg_mgr.log_info('===Rank-1 (Include identical-view cases)===')
        msg_mgr.log_info('NM: %.3f ' % (np.mean(acc[0, :, :, 0])))
        msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
        msg_mgr.log_info('NM: %.3f ' % (de_diag(acc[0, :, :, 0])))
        result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, 0])
    else:
        # msg_mgr.log_info('===Rank-1 (Include identical-view cases)===')
        msg_mgr.log_info('Rank-1: %.3f ' % (np.mean(acc[0, :, :, 0])))
        msg_mgr.log_info('Rank-2: %.3f ' % (np.mean(acc[0, :, :, 1])))
        msg_mgr.log_info('Rank-3: %.3f ' % (np.mean(acc[0, :, :, 2])))
        msg_mgr.log_info('Rank-4: %.3f ' % (np.mean(acc[0, :, :, 3])))
        msg_mgr.log_info('Rank-5: %.3f ' % (np.mean(acc[0, :, :, 4])))
        result_dict["scalar/test_accuracy/NM"] = np.mean(acc[0, :, :, 0])
        # msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
        # msg_mgr.log_info('NM: %.3f ' % (de_diag(acc[0, :, :, 0])))
        # msg_mgr.log_info('NM: {}'.format(de_diag(acc[0, :, :, 0], True)))
        # result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, 0])
    return result_dict









def identification_fvg(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()

    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    # print(label.len())
    # print(seq_type.len())
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    # sample_num = len(feature)

    probe_seq_dict = {'CASIA-B': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                     'OUMVLP': [['00']],
                     'FVG_CCVID_WS': [['1_04', '1_05', '1_06', '1_07', '1_08', '1_09', '2_04', '2_05', '2_06']],
                     'FVG_CCVID_BGHT':[['1_10', '1_11', '1_12']],
                     'FVG_CCVID_CL':[['2_07', '2_08', '2_09']],
                     'FVG_CCVID_MP':[['2_10', '2_11', '2_12']],
                     'FVG_CCVID_ALL':[['1_01', '1_03', '1_04', '1_05', '1_06', '1_07', '1_08', '1_09', '1_10', '1_11', '1_12',
                                       '2_01', '2_03', '2_04', '2_05', '2_06', '2_07', '2_08', '2_09', '2_10', '2_11', '2_12',
                                       '3_01', '3_02', '3_03', '3_04', '3_05', '3_06', '3_07', '3_08', '3_09', '3_10', '3_11', '3_12']],
                     'CCVID_CC':[['1_01', '1_02', '1_03',
                               '2_01', '2_02', '2_03', '2_04', '2_05', '2_06', '2_10', '2_11', '2_12',
                               '3_01', '3_02', '3_03', '3_04', '3_05', '3_06', '3_07', '3_08', '3_09', '3_10', '3_11', '3_12']],
                     }
    gallery_seq_dict = {'CASIA-B': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']],
                        'FVG_CCVID_WS':[['1_02', '2_02']],
                        'FVG_CCVID_BGHT':[['1_02']],
                        'FVG_CCVID_CL':[['2_02']],
                        'FVG_CCVID_MP':[['2_02']],
                        'FVG_CCVID_ALL':[['1_02', '2_02']],
                        'CCVID_CC':[['1_04', '1_05', '1_06', '1_07', '1_08', '1_09', '1_10', '1_11', '1_12',
                                  '2_07', '2_08', '2_09',
                                  '1_01_3', '1_02_3', '1_03_3', '1_04_3', '1_05_3', '1_06_3', '1_07_3', '1_08_3', '1_09_3', '1_10_3', '1_11_3', '1_12_3']]
                        }
    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)

    for dataset in ['FVG_CCVID_WS', 'FVG_CCVID_BGHT', 'FVG_CCVID_CL', 'FVG_CCVID_MP', 'FVG_CCVID_ALL']:
        print(dataset)
        num_rank = 5
        acc = np.zeros([len(probe_seq_dict[dataset]),
                        view_num, view_num, num_rank]) - 1.
        for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
            for gallery_seq in gallery_seq_dict[dataset]:
                for (v1, probe_view) in enumerate(view_list):
                    for (v2, gallery_view) in enumerate(view_list):
                        gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                            view, [gallery_view])
                        gallery_x = feature[gseq_mask, :]
                        gallery_y = label[gseq_mask]

                        pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                            view, [probe_view])
                        probe_x = feature[pseq_mask, :]
                        probe_y = label[pseq_mask]

                        dist = cuda_dist(probe_x, gallery_x, metric)
                        idx = dist.sort(1)[1].cpu().numpy()
                        acc[p, v1, v2, :] = np.round(
                            np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                                0) * 100 / dist.shape[0], 2)
        result_dict = {}
        if 'CASIA-B' in dataset:
            for i in range(1):
                msg_mgr.log_info(
                    '===Rank-%d (Include identical-view cases)===' % (i + 1))
                msg_mgr.log_info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                    np.mean(acc[0, :, :, i]),
                    np.mean(acc[1, :, :, i]),
                    np.mean(acc[2, :, :, i])))
            for i in range(1):
                msg_mgr.log_info(
                    '===Rank-%d (Exclude identical-view cases)===' % (i + 1))
                msg_mgr.log_info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                    de_diag(acc[0, :, :, i]),
                    de_diag(acc[1, :, :, i]),
                    de_diag(acc[2, :, :, i])))
            result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, i])
            result_dict["scalar/test_accuracy/BG"] = de_diag(acc[1, :, :, i])
            result_dict["scalar/test_accuracy/CL"] = de_diag(acc[2, :, :, i])
            np.set_printoptions(precision=2, floatmode='fixed')
            for i in range(1):
                msg_mgr.log_info(
                    '===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
                msg_mgr.log_info('NM: {}'.format(de_diag(acc[0, :, :, i], True)))
                msg_mgr.log_info('BG: {}'.format(de_diag(acc[1, :, :, i], True)))
                msg_mgr.log_info('CL: {}'.format(de_diag(acc[2, :, :, i], True)))
        elif 'OUMVLP' in dataset:
            msg_mgr.log_info('===Rank-1 (Include identical-view cases)===')
            msg_mgr.log_info('NM: %.3f ' % (np.mean(acc[0, :, :, 0])))
            msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
            msg_mgr.log_info('NM: %.3f ' % (de_diag(acc[0, :, :, 0])))
            result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, 0])
        else:
            # msg_mgr.log_info('===Rank-1 (Include identical-view cases)===')
            msg_mgr.log_info('Rank-1: %.3f ' % (np.mean(acc[0, :, :, 0])))
            msg_mgr.log_info('Rank-2: %.3f ' % (np.mean(acc[0, :, :, 1])))
            msg_mgr.log_info('Rank-3: %.3f ' % (np.mean(acc[0, :, :, 2])))
            msg_mgr.log_info('Rank-4: %.3f ' % (np.mean(acc[0, :, :, 3])))
            msg_mgr.log_info('Rank-5: %.3f ' % (np.mean(acc[0, :, :, 4])))
            result_dict["scalar/test_accuracy/NM"] = np.mean(acc[0, :, :, 0])
            # msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
            # msg_mgr.log_info('NM: %.3f ' % (de_diag(acc[0, :, :, 0])))
            # msg_mgr.log_info('NM: {}'.format(de_diag(acc[0, :, :, 0], True)))
            # result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, 0])
    return result_dict
















def identification_real_scene(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    feature, label, seq_type = data['embeddings'], data['labels'], data['types']
    label = np.array(label)

    gallery_seq_type = {'0001-1000': ['1', '2'],
                        "HID2021": ['0'], '0001-1000-test': ['0'],
                        'GREW': ['01'], 'TTG-200': ['1']}
    probe_seq_type = {'0001-1000': ['3', '4', '5', '6'],
                      "HID2021": ['1'], '0001-1000-test': ['1'],
                      'GREW': ['02'], 'TTG-200': ['2', '3', '4', '5', '6']}

    num_rank = 20
    acc = np.zeros([num_rank]) - 1.
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = label[pseq_mask]

    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.cpu().sort(1)[1].numpy()
    acc = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                          0) * 100 / dist.shape[0], 2)
    msg_mgr.log_info('==Rank-1==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[0])))
    msg_mgr.log_info('==Rank-5==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[4])))
    msg_mgr.log_info('==Rank-10==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[9])))
    msg_mgr.log_info('==Rank-20==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[19])))
    return {"scalar/test_accuracy/Rank-1": np.mean(acc[0]), "scalar/test_accuracy/Rank-5": np.mean(acc[4])}


def identification_GREW_submission(data, dataset, metric='euc'):
    get_msg_mgr().log_info("Evaluating GREW")
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view = np.array(view)
    gallery_seq_type = {'GREW': ['01', '02']}
    probe_seq_type = {'GREW': ['03']}
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = view[pseq_mask]

    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.cpu().sort(1)[1].numpy()

    save_path = os.path.join(
        "GREW_result/"+strftime('%Y-%m%d-%H%M%S', localtime())+".csv")
    mkdir("GREW_result")
    with open(save_path, "w") as f:
        f.write("videoId,rank1,rank2,rank3,rank4,rank5,rank6,rank7,rank8,rank9,rank10,rank11,rank12,rank13,rank14,rank15,rank16,rank17,rank18,rank19,rank20\n")
        for i in range(len(idx)):
            r_format = [int(idx) for idx in gallery_y[idx[i, 0:20]]]
            output_row = '{}'+',{}'*20+'\n'
            f.write(output_row.format(probe_y[i], *r_format))
        print("GREW result saved to {}/{}".format(os.getcwd(), save_path))
    return


def evaluate_HID(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info("Evaluating HID")
    feature, label, seq_type = data['embeddings'], data['labels'], data['types']
    label = np.array(label)
    seq_type = np.array(seq_type)
    probe_mask = (label == "probe")
    gallery_mask = (label != "probe")
    gallery_x = feature[gallery_mask, :]
    gallery_y = label[gallery_mask]
    probe_x = feature[probe_mask, :]
    probe_y = seq_type[probe_mask]

    feat = np.concatenate([probe_x, gallery_x])
    dist = cuda_dist(feat, feat, metric).cpu().numpy()
    msg_mgr.log_info("Starting Re-ranking")
    re_rank = re_ranking(dist, probe_x.shape[0], k1=6, k2=6, lambda_value=0.3)
    idx = np.argsort(re_rank, axis=1)

    save_path = os.path.join(
        "HID_result/"+strftime('%Y-%m%d-%H%M%S', localtime())+".csv")
    mkdir("HID_result")
    with open(save_path, "w") as f:
        f.write("videoID,label\n")
        for i in range(len(idx)):
            f.write("{},{}\n".format(probe_y[i], gallery_y[idx[i, 0]]))
        print("HID result saved to {}/{}".format(os.getcwd(), save_path))
    return


def re_ranking(original_dist, query_num, k1, k2, lambda_value):
    # Modified from https://github.com/michuanhaohao/reid-strong-baseline/blob/master/utils/re_ranking.py
    all_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(
                np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                            :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + \
        original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def mean_iou(data, dataset):
    labels = data['mask']
    pred = data['pred']
    miou = MeanIOU(pred, labels)
    get_msg_mgr().log_info('mIOU: %.3f' % (miou.mean()))
    return {"scalar/test_accuracy/mIOU": miou}
