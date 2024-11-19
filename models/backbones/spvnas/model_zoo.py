import os
import sys
import torch
from torchpack import distributed as dist
sys.path.append(os.path.join(os.path.dirname(__file__)))
from core.models.semantic_kitti.spvcnn import SPVCNN
#该代码定义了一个名为 spvcnn 的函数，用于创建和初始化一个 SPVCNN 模型实例。此模型是一种基于稀疏卷积的神经网络，通常用于处理点云数据或稀疏3D数据。函数返回一个准备好在 GPU 或 CPU 上运行的模型实例
__all__ = ['spvcnn'] #表示当前模块中唯一对外公开的接口是 spvcnn 函数。其他内容即使定义在该模块中，也不会被外部直接访问


def spvcnn(output_dim=16): #output_dim（默认值为 16）：指定 SPVCNN 的输出类别数，即分类任务中的类别数量或其他任务中的输出特征维度

    model = SPVCNN(
        num_classes=output_dim, # num_classes：模型的输出类别数
        cr=0.64, #cr：通道缩放比例（Channel Reduction Ratio），用以调整模型的通道数，从而控制网络复杂度
        pres=0.05, #pres：点云数据的输入分辨率（Point Cloud Resolution），即体素化的空间分辨率
        vres=0.05 #vres：体素分辨率（Voxel Resolution），即体素网格的大小
    ).to('cuda:%d' % dist.local_rank() if torch.cuda.is_available() else 'cpu')
    #检查是否有可用的 GPU 若有 GPU 可用，则将模型分配到当前进程的 GPU 上（通常适用于分布式训练） 若无 GPU，则将模型分配到 CPU

    return model
