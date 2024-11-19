import os
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.aggregators.SOP import *
from models.backbones.spvnas.model_zoo import spvcnn
from models.pipelines.pipeline_utils import *
###LOGG3D 模型能够有效地从点云数据中提取空间特征并进行聚合，进而生成全局或局部的特征表示
__all__ = ['LOGG3D']  #指定当 from 模块 import * 时，模块中仅导出 LOGG3D 类
#_all__ 变量是一个字符串列表，包含了你希望从模块中暴露给外部的所有名称。如果在模块中定义了 __all__，只有在 __all__ 中列出的对象才能通过 from module import * 被导入。如果没有定义 __all__，那么默认情况下，所有不以下划线 (_) 开头的名称都会被导入

#该类继承自 nn.Module，实现一个 3D 点云特征提取模型，结合了稀疏卷积网络和特征池化操作
class LOGG3D(nn.Module): #LOGG3D 类是一个深度学习模型，用于处理和分析 3D 点云数据
    def __init__(self, feature_dim=16):
        super(LOGG3D, self).__init__()

        self.spvcnn = spvcnn(output_dim=feature_dim) # # 稀疏体素卷积网络，用于提取点云特征。 output_dim=feature_dim制定了输出的特征维度是16，即点云中的每个点或每个区域的特征向量的大小
        self.sop = SOP(
            signed_sqrt=False, do_fc=False, input_dim=feature_dim, is_tuple=False)#使用池化操作 (SOP) 将点云的特征进行降维和聚合，池化操作用于减少特征的维度或大小，同时保留关键信息
        #signed_sqrt=False: 不使用有符号平方根操作 do_fc=False: 不使用全连接层，因此输出的特征将直接进行后续处理，而不会经过额外的全连接层 input_dim=feature_dim: 输入特征维度为 feature_dim
        #is_tuple=False：表示 sop 模块是否处理元组类型的数据。在这里，它被设置为 False，意味着 sop 不会处理元组，而是处理普通的张量数据

    def forward(self, x):
        _, counts = torch.unique(x.C[:, -1], return_counts=True) #torch.unique(x.C[:, -1], return_counts=True)获取点云中不同帧（批次）的数量 (counts)
#统计不同帧的数量，x.C表示一个包含点云数据的张量，其中每一行可能是一个点的信息，最后一列是每个点所在的帧的标识
        #唯一的帧标识 同时返回每个唯一标识出现的次数，就是每一帧包含多少个点
        x = self.spvcnn(x) #self.spvcnn(x) 使用稀疏卷积提取点云的空间特征  稀疏卷积：高效地提取空间信息，同时保持较低的计算复杂度
        y = torch.split(x, list(counts)) #torch.split(x, list(counts)) 按批次数量将特征划分为帧对应的分组，根据点的数量拆分成多个子张量
        x = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 0, 2) #torch.nn.utils.rnn.pad_sequence 将每帧的特征序列填充为等长数组  将前面按数量划分的不同长度的张量填充成一个长度相同的序列
       #torch.nn.utils.rnn.pad_sequence(list(y)) 会将一个由不同长度的张量组成的列表（y）填充成一个长度相同的序列
        #pad_sequence 的作用是将这些张量填充到相同的长度
        #permute(1, 0, 2) 对填充后的张量进行转置操作  原始张量的维度是 (batch_size, sequence_length, feature_size)，permute(1, 0, 2) 会将其转换为 (sequence_length, batch_size, feature_size)
        
        x = self.sop(x) #self.sop(x) 将填充后的特征进一步聚合为全局特征  #特征聚合操作，对整个输入序列的特征进行聚合，生成一个全局特征表示
        return x, y[:2] #全局聚合的特征   前两帧的特征分组
        #x 是经过一系列处理后的张量，它表示 全局特征，通常用于总结整个输入点云或一组输入数据的全局信息
#返回前两帧的特征可能是因为需要局部特征的细节，或者前两帧特帧在任务中具有较高的相关性

#这里是只是处理一个点云文件，如果要处理多个点云文件的话，是按批次来的
if __name__ == '__main__':
    _backbone_model_dir = os.path.join(
        os.path.dirname(__file__), '../backbones/spvnas')
    sys.path.append(_backbone_model_dir) #将存储模型和点云数据的路径拼接成绝对路径
    lidar_pc = np.fromfile(_backbone_model_dir +
                           '/tutorial_data/000000.bin', dtype=np.float32)  #点云文件 000000.bin 是一个 .bin 格式文件，存储了 3D 点云 (x, y, z, intensity)
    lidar_pc = lidar_pc.reshape(-1, 4) #展平并组织成二维数组，-1是自动计算行数 就是计算点云的点数
    input = make_sparse_tensor(lidar_pc, 0.05).cuda() #使用 make_sparse_tensor(lidar_pc, 0.05) 将点云数据体素化，生成稀疏张量，降低计算复杂度，.cuda() 将稀疏张量转移到 GPU 上，以便加速后续的计算

    model = LOGG3D().cuda() #创建并初始化一个 LOGG3D 模型实例，同时将模型加载到 GPU
    model.train() #调用 model.train() 将模型切换到训练模式
    output = model(input) #传入模型进行前向传播
    print('output size: ', output[0].size()) #输出模型处理后的全局特征 (output[0]) 的维度信息




#张量是一个多维数组，类似于矩阵或者向量，一维张量是向量，二维张量是矩阵
