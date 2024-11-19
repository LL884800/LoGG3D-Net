import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

#STN3d（Spatial Transformer Network for 3D）是一个3D空间变换模块。其主要功能是学习输入点云的空间变换矩阵，用以对点云进行对齐或标准化，从而提高模型对旋转和平移变化的鲁棒性

class STN3d(nn.Module):
    def __init__(self, num_points=2500, k=3, use_bn=True): #输入点云的点数，默认为 2500
        super(STN3d, self).__init__()
        self.k = k #空间变换矩阵的大小，默认为 3（表示3×3变换矩阵）
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn #是否在网络中使用批量归一化（Batch Normalization），默认启用

        #卷积层
        #第一层将输入点特征映射到64维，第二层映射到128维，第三层映射到1024维
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1, 1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1, 1))
        #池化层（mp1） 使用全局最大池化（MaxPool2d），将特征维度缩减为 1024
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        #全连接层（fc1, fc2, fc3） 用于将特征压缩并最终预测 k×k 的空间变换矩阵 fc3 的初始权重和偏置设为零，保证初始输出为单位矩阵
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            #批量归一化层 若启用批量归一化，则对每一层的输出进行归一化以加速训练和提高稳定性
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(
            1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
"""
输入张量 x 通过卷积层和池化层提取全局特征
特征通过全连接层生成变换矩阵 x
加入单位矩阵（iden）作为初始化偏移，以确保网络稳定训练
输出最终的 k×k 变换矩阵
"""
#PointNetfeat 是 PointNet 中的特征提取模块，用于从点云中提取全局和局部特征。该模块可以选择输出全局特征或每个点的局部特征
class PointNetfeat(nn.Module): #是否返回全局特征（默认返回全局特征） 是否对特征进行二次空间变换 是否在最后执行全局最大池化
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points, k=3, use_bn=False) #使用 STN3d 学习点云的3D变换和特征变换
        self.feature_trans = STN3d(num_points=num_points, k=64, use_bn=False)
        self.apply_feature_trans = feature_transform
        #conv1 将点从3维映射到64维。后续卷积层逐步提取更深层次的特征（128维到1024维）
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv3 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv4 = torch.nn.Conv2d(64, 128, (1, 1))
        self.conv5 = torch.nn.Conv2d(128, 1024, (1, 1))
        #批量归一化层
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        #池化层
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = torch.matmul(torch.squeeze(x), trans)
        x = x.view(batchsize, 1, -1, 3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x
        if self.apply_feature_trans:
            f_trans = self.feature_trans(x)
            x = torch.squeeze(x)
            if batchsize == 1:
                x = torch.unsqueeze(x, 0)
            x = torch.matmul(x.transpose(1, 2), f_trans)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batchsize, 64, -1, 1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        if not self.max_pool:
            return x
        else:
            x = self.mp1(x)
            x = x.view(-1, 1024)
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans



"""
输入点云张量 𝑥维度为 (𝐵,𝑁,3)，即批量大小为 B，点数为 N，每个点有3个坐标。
输出：
如果 global_feat=True，输出形状为 (B,1024)。
如果 global_feat=False，输出形状为 (𝐵,1088,𝑁)
(B,1088,N)，其中 1088 是 1024（全局特征）加上 64（局部特征）
"""
