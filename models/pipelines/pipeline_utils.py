import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate


def make_sparse_tensor(lidar_pc, voxel_size=0.05, return_points=False):
    #该函数的主要功能是将原始点云数据转换为稀疏张量 (SparseTensor)，用于稀疏卷积网络的输入
    # get rounded coordinates
    coords = np.round(lidar_pc[:, :3] / voxel_size) #lidar_pc 输入的 LiDAR 点云数据，形状为 (N, M)，其中 N 是点的数量，M 是点的特征维度（通常包括 x, y, z, intensity 等）
    coords -= coords.min(0, keepdims=1)
    feats = lidar_pc

    # sparse quantization: filter out duplicate points
    _, indices = sparse_quantize(coords, return_index=True) #sparse_quantize 函数对体素坐标进行量化，去掉重复的点，仅保留每个体素中的一个点
    coords = coords[indices]
    feats = feats[indices]  #indices 保存了被保留点的索引，用于更新 coords 和原始点的特征 (feats)

    # construct the sparse tensor
    inputs = SparseTensor(feats, coords) #SparseTensor(feats, coords) 将点的特征 (feats) 和体素坐标 (coords) 组合为稀疏张量
    inputs = sparse_collate([inputs]) #sparse_collate 将输入张量批处理成符合网络输入格式的张量
    inputs.C = inputs.C.int() #将稀疏张量的坐标 (inputs.C) 转换为整数类型
    if return_points:
        return inputs, feats
    else:
        return inputs

#输入：点云数据和体素大小，输出：稀疏张量，用于稀疏卷积网络
#功能：对点云数据进行体素化和去冗余处理
