import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate


def make_sparse_tensor(lidar_pc, voxel_size=0.05, return_points=False):
    #该函数的主要功能是将原始点云数据转换为稀疏张量 (SparseTensor)，用于稀疏卷积网络的输入
    # get rounded coordinates
    coords = np.round(lidar_pc[:, :3] / voxel_size) #lidar_pc 输入的 LiDAR 点云数据，形状为 (N, M)，其中 N 是点的数量，M 是点的特征维度（通常包括 x, y, z, intensity 等）
    #将3d坐标转换成体素网络的坐标。round四舍五入  lidar_pc[:, :3]切片操作，只选择所有行的前3列，忽略强度值
    coords -= coords.min(0, keepdims=1) #keepdims=1 使得输出保持原来的维度  //对每一个体素化之后的坐标减去对应的最小值，使得原点坐标是（0，0，0）
    feats = lidar_pc #包含原始点云数据，四个维度，这里有强度信息

    # sparse quantization: filter out duplicate points
    _, indices = sparse_quantize(coords, return_index=True) #sparse_quantize 函数对体素坐标进行量化，去掉重复的点，仅保留每个体素中的一个点
   #return_index=True 表示函数还会返回一个 indices，它保存了哪些点被保留下来的索引
    #_, indices 表示我们只关心 indices，不关心返回的第一个值
    coords = coords[indices] #仅仅保存筛选之后的点云坐标
    feats = feats[indices]  #indices 保存了被保留点的索引，用于更新 coords 和原始点的特征 (feats)

    # construct the sparse tensor
    inputs = SparseTensor(feats, coords) #SparseTensor(feats, coords) 将点的特征 (feats) 和体素坐标 (coords) 组合为稀疏张量
    inputs = sparse_collate([inputs]) #sparse_collate 将输入张量批处理成符合网络输入格式的张量
    #sparse_collate 是一个用于批处理稀疏张量的函数。它会将多个稀疏张量（在这里是一个列表，只有一个元素 inputs）合并为一个批次
    #这个函数的作用是将点云数据整理成适合传入神经网络的格式，确保张量的形状符合网络的输入要求
    inputs.C = inputs.C.int() #将稀疏张量的坐标 (inputs.C) 转换为整数类型
    if return_points:  #根据return_points来决定是返回稀疏张量还是原始点云
        return inputs, feats
    else:
        return inputs

#输入：点云数据和体素大小，输出：稀疏张量，用于稀疏卷积网络
#功能：对点云数据进行体素化和去冗余处理
