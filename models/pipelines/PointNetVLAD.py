import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.aggregators.NetVLAD import *
from models.backbones.PointNet import *

class PointNetVLAD(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True, output_dim=1024):
        super(PointNetVLAD, self).__init__()
        self.point_net = PointNetfeat(num_points=num_points, global_feat=global_feat,
                                      feature_transform=feature_transform, max_pool=max_pool)
        #点云中包含的点数量 global_feat: 是否提取全局特征 feature_transform: 是否对特征空间进行变换 max_pool: 是否在 PointNet 的输出层使用最大池化操作
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=num_points, cluster_size=64,
                                     output_dim=output_dim, gating=True, add_batch_norm=True,
                                     is_training=True)
#feature_size=1024: 输入特征的维度（与 PointNetfeat 输出保持一致） max_samples=num_points: 每帧点云的最大样本数
    #cluster_size=64: 聚类中心数量 output_dim=output_dim: 输出描述符的维度 gating=True: 是否使用门控机制
    #add_batch_norm=True: 是否添加批归一化 is_training=True: 是否为训练模式
    def forward(self, x):
        x = self.point_net(x) #使用 PointNetfeat 提取点云特征（可能是局部或全局特征）
        x = self.net_vlad(x) #使用 NetVLADLoupe 聚合特征，将其转化为固定长度的全局描述符
        return x


if __name__ == '__main__':
    num_points = 4096
    sim_data = Variable(torch.rand(44, 1, num_points, 3)) #sim_data 是一个模拟的点云数据，尺寸为 (44, 1, 4096, 3) 表示 44 帧点云，每帧点云包含 4096 个点，每个点有 3 个坐标值 (x, y, z)

    pnv = PointNetVLAD(global_feat=True, feature_transform=True,
                       max_pool=False, output_dim=256, num_points=num_points)  # .cuda()
   #全局特征 (global_feat=True) 特征变换 (feature_transform=True) 输出维度为 256 (output_dim=256)
    
    pnv.train()
    out = pnv(sim_data)
    print('pnv', out.size())
