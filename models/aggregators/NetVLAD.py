import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
#用于对特征进行聚类和生成全局描述符（通常用于点云特征提取、场景识别等）

class NetVLADLoupe(nn.Module): #负责实现 VLAD（Vector of Locally Aggregated Descriptors）聚类及后续的特征处理
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size #输入特征的维度
        self.max_samples = max_samples #每次输入的最大样本数
        self.output_dim = output_dim   #输出特征的维度
        self.is_training = is_training  #是否为训练模式（未显式使用，可能用于调试）
        self.gating = gating      #是否启用上下文门控机制
        self.add_batch_norm = add_batch_norm   #是否使用批量归一化
        self.cluster_size = cluster_size  #聚类中心的数量
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size)) #用于将输入特征映射到聚类中心的权重矩阵
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))  #用于计算 VLAD 偏移
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size)) #用于将特征转换为指定的输出维度

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):  #前向传播方法 forward
        x = x.transpose(1, 3).contiguous()  # 调整输入维度顺序
        x = x.view((-1, self.max_samples, self.feature_size))
        activation = torch.matmul(x, self.cluster_weights) #转置输入张量，并调整形状为 (-1, max_samples, feature_size)
        #计算聚类激活值 输入特征与聚类权重点乘，得到每个样本的聚类激活值
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1,
                                         self.max_samples, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        #根据配置对聚类激活值进行归一化处理或加偏置
        activation = self.softmax(activation)
        activation = activation.view((-1, self.max_samples, self.cluster_size))
        #
        #使用 Softmax 归一化激活值，使其成为概率分布
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2
        #聚类激活值求和后，与权重 cluster_weights2 相乘得到偏移量 a
        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a
        
        #特征点乘激活值，并减去偏移量，得到聚类后的 VLAD 特征
        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.reshape((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)
        #对 VLAD 特征执行两次 L2 标准化操作
        vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)
        #使用 hidden1_weights 映射到输出维度，并批量归一化
        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad

#使用门控机制调整激活值，增强重要特征的表示能力
class GatingContext(nn.Module): #实现上下文门控机制，进一步调整特征的激活
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation
