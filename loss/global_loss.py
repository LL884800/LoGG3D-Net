# Based on: https://github.com/cattaneod/PointNetVlad-Pytorch
import numpy as np
import torch

#函数用于计算查询样本（query）与正样本集（pos_vecs）之间的最大距离
def best_pos_distance(query, pos_vecs):
    num_pos = pos_vecs.shape[0]
    query_copies = query.repeat(int(num_pos), 1) #查询向量 query 进行复制，使其与正样本集 pos_vecs 的每个样本进行配对
    # ((pos_vecs - query_copies) ** 2).sum(2)
    diff = ((pos_vecs - query_copies)**2).sum(1)  #每个正样本与查询样本之间的平方欧氏距离
    max_pos_dist = diff.max()
    return max_pos_dist #返回正样本中与查询样本的最大距离

#三元组损失用于训练模型
def triplet_loss(tuple_vecs, config):
    tuple_vecs = torch.split(
        tuple_vecs, [1, config.positives_per_query, config.negatives_per_query])
    #查询向量（q_vec）、正样本向量（pos_vecs）和负样本向量（neg_vecs）
    q_vec, pos_vecs, neg_vecs = tuple_vecs[0], tuple_vecs[1], tuple_vecs[2]
    max_pos_dist = best_pos_distance(q_vec, pos_vecs)

    num_neg = neg_vecs.shape[0]
    query_copies = q_vec.repeat(int(num_neg), 1)
    max_pos_dist = max_pos_dist.view(1, -1)
    max_pos_dist = max_pos_dist.repeat(int(num_neg), 1)

    neg_dists = ((neg_vecs - query_copies) ** 2).sum(1)
    loss = config.loss_margin_1 + max_pos_dist - \
        neg_dists.reshape((int(num_neg), 1))
    #计算损失 loss，它是查询样本与负样本之间距离与最大正样本距离之差。然后对损失进行裁剪，确保其最小值为0
    
    loss = loss.clamp(min=0.0)
    #如果配置 lazy_loss 为 True，则损失取最大值；否则，损失取所有负样本损失的和
    if config.lazy_loss:
        triplet_loss = loss.max()
    else:
        triplet_loss = loss.sum()

    return triplet_loss

#四元组损失是三元组损失的扩展，除了正样本和负样本外，还考虑了额外的负样本other_neg
def quadruplet_loss(tuple_vecs, config):
    tuple_vecs = torch.split(
        tuple_vecs, [1, config.positives_per_query, config.negatives_per_query, 1])
    q_vec, pos_vecs, neg_vecs, other_neg = tuple_vecs[0], tuple_vecs[1], tuple_vecs[2], tuple_vecs[3]
    #输入的 tuple_vecs 包含查询向量（q_vec）、正样本向量（pos_vecs）、负样本向量（neg_vecs）和其他负样本向量（other_neg）
    max_pos_dist = best_pos_distance(q_vec, pos_vecs)

    num_neg = neg_vecs.shape[0]
    query_copies = q_vec.repeat(int(num_neg), 1)
    max_pos_dist = max_pos_dist.view(1, -1)
    max_pos_dist = max_pos_dist.repeat(int(num_neg), 1)

    neg_dists = ((neg_vecs - query_copies) ** 2).sum(1)
    loss = config.loss_margin_1 + max_pos_dist - \
        neg_dists.reshape((int(num_neg), 1))
    loss = loss.clamp(min=0.0) #clamp(min=0.0)：这是为了确保损失不为负值，因为我们希望最小损失为0，表示正样本和负样本的距离已经足够分开
    #如果配置 lazy_loss 为 True，则第二个损失取最大值；否则，取所有第二负样本损失的和。
    if config.lazy_loss:
        triplet_loss = loss.max()
    else:
        triplet_loss = loss.sum()

    other_neg_copies = other_neg.repeat(int(num_neg), 1)
    other_neg_dists = ((neg_vecs - other_neg_copies) ** 2).sum(1)
    second_loss = config.loss_margin_2 + max_pos_dist - \
        other_neg_dists.reshape((int(num_neg), 1))

    second_loss = second_loss.clamp(min=0.0)
    if config.lazy_loss:
        second_loss = second_loss.max()
    else:
        second_loss = second_loss.sum()

    total_loss = triplet_loss + second_loss
    return total_loss
