import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.misc_utils import pdist, hashM
//点对比损失函数，用于神经网络的训练

def point_contrastive_loss(F0, F1, positive_pairs, config,
                           num_pos=5192,
                           num_hn_samples=2048):
                #F0, F1：表示两组点的特征矩阵（例如，点云的描述符或特征图），positive_pairs：正样本对，config：配置对象，num_pos：要随机选择的正样本对的数量，num_hn_samples：从负样本中选择的最难样本的数量（即，每个正样本对选择的最难负样本的数量）
    """
    Randomly select 'num-pos' positive pairs. 
    Find the hardest-negative (from a random subset of num_hn_samples) for each point in a positive pair.
    Calculate contrastive loss on the tuple (p1,p2,hn1,hn2)
    Based on: https://github.com/chrischoy/FCGF/blob/master/lib/trainer.py
    """

                             #首先，从 F0 和 F1 中随机选择 num_hn_samples 个样本
    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)
#如果 N_pos_pairs（正样本对的数量）大于 num_pos，则从正样本对中随机选择 num_pos 对
    if N_pos_pairs > num_pos:
        pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
        sample_pos_pairs = positive_pairs[pos_sel]
    else:
        sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0]  # .long()
    pos_ind1 = sample_pos_pairs[:, 1]  # .long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]
#对于每个正样本对，计算 F0 和 F1 特征之间的 L2 距离，分别计算 F0 -> F1 和 F1 -> F0 的距离
    D01 = pdist(posF0, subF1, dist_type='L2')
    D10 = pdist(posF1, subF0, dist_type='L2')

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
        positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = hashM(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = hashM([pos_ind0, D01ind], hash_seed)
    neg_keys1 = hashM([D10ind, pos_ind1], hash_seed)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - config.point_pos_margin)
    neg_loss0 = F.relu(config.point_neg_margin - D01min[mask0]).pow(2)
    neg_loss1 = F.relu(config.point_neg_margin - D10min[mask1]).pow(2)

    pos_loss = pos_loss.mean()
    neg_loss = (neg_loss0.mean() + neg_loss1.mean()) / 2
 # 最后，正样本损失和负样本损失的加权和作为最终的损失，负样本损失会乘以 config.point_neg_weight
    loss = pos_loss + config.point_neg_weight * neg_loss
    return loss


def point_infonce_loss(query_feats, pos_feats, pos_pairs, neg_pairs, config):  # TODO
    return 0
