import os
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.aggregators.SOP import *
from models.backbones.spvnas.model_zoo import spvcnn
from models.pipelines.pipeline_utils import *

__all__ = ['LOGG3D']  #指定当 from 模块 import * 时，模块中仅导出 LOGG3D 类

#该类继承自 nn.Module，实现一个 3D 点云特征提取模型，结合了稀疏卷积网络和特征池化操作
class LOGG3D(nn.Module):
    def __init__(self, feature_dim=16):
        super(LOGG3D, self).__init__()

        self.spvcnn = spvcnn(output_dim=feature_dim) # # 稀疏体素卷积网络，用于提取点云特征。
        self.sop = SOP(
            signed_sqrt=False, do_fc=False, input_dim=feature_dim, is_tuple=False)#使用池化操作 (SOP) 将点云的特征进行降维和聚合
        #signed_sqrt=False: 不使用有符号平方根操作 do_fc=False: 不使用全连接层 input_dim=feature_dim: 输入特征维度为 feature_dim

    def forward(self, x):
        _, counts = torch.unique(x.C[:, -1], return_counts=True) #torch.unique(x.C[:, -1], return_counts=True)获取点云中不同帧（批次）的数量 (counts)

        x = self.spvcnn(x) #self.spvcnn(x) 使用稀疏卷积提取点云的空间特征
        y = torch.split(x, list(counts)) #torch.split(x, list(counts)) 按批次数量将特征划分为帧对应的分组
        x = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 0, 2) #torch.nn.utils.rnn.pad_sequence 将每帧的特征序列填充为等长数组
        x = self.sop(x) #self.sop(x) 将填充后的特征进一步聚合为全局特征
        return x, y[:2] #全局聚合的特征   前两帧的特征分组


if __name__ == '__main__':
    _backbone_model_dir = os.path.join(
        os.path.dirname(__file__), '../backbones/spvnas')
    sys.path.append(_backbone_model_dir)
    lidar_pc = np.fromfile(_backbone_model_dir +
                           '/tutorial_data/000000.bin', dtype=np.float32)  #点云文件 000000.bin 是一个 .bin 格式文件，存储了 3D 点云 (x, y, z, intensity)
    lidar_pc = lidar_pc.reshape(-1, 4)
    input = make_sparse_tensor(lidar_pc, 0.05).cuda() #使用 make_sparse_tensor(lidar_pc, 0.05) 将点云数据体素化，生成稀疏张量

    model = LOGG3D().cuda()
    model.train()
    output = model(input)
    print('output size: ', output[0].size()) #输出模型处理后的全局特征 (output[0]) 的维度信息
