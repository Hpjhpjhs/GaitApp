import torch
import numpy as np
import torch.nn.functional as F
from utils import get_msg_mgr


def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=2)  # n p c
        y = F.normalize(y, p=2, dim=2)  # n p c
    num_bin = x.size(1)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, i, ...]
        _y = y[:, i, ...]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                1).transpose(0, 1) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
            # dist = 1 - torch.exp((dist)*(-0.6))
            # dist = 1- 1 / (1 + dist)
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



def identification_ccvid(data, dataset, metric='euc'):
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

    for dataset in ['CCVID_CC']:
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



def identification_casiab(data, dataset, metric='euc'):
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
    return result_dict







def identification_real_scene(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    feature, label, seq_type = data['embeddings'], data['labels'], data['types']
    label = np.array(label)

    gallery_seq_type = {'0001-1000': ['1', '2'],
                        "HID2021": ['0'], '0001-1000-test': ['0']}
    probe_seq_type = {'0001-1000': ['3', '4', '5', '6'],
                      "HID2021": ['1'], '0001-1000-test': ['1']}

    num_rank = 5
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
    return {"scalar/test_accuracy/Rank-1": np.mean(acc[0]), "scalar/test_accuracy/Rank-5": np.mean(acc[4])}
