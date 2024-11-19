import torch
import torchsparse.nn.functional as F
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets
#Python 中的一个模块定义，指定了模块公开的接口，即导入时可以直接使用的函数列表：initial_voxelize、point_to_voxel 和 voxel_to_point
__all__ = ['initial_voxelize', 'point_to_voxel', 'voxel_to_point']


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):  #将一个 PointTensor 数据进行初始体素化操作
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1)
    #z: 输入的点云数据，类型为 PointTensor，包含坐标 (C) 和特征 (F)，init_res: 初始分辨率，通常用于对点云坐标进行归一化，after_res: 体素化后的目标分辨率
    

    pc_hash = F.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash) # 使用 torch.floor(new_float_coord) 取整后，计算点云的哈希值 pc_hash，并从中生成唯一的稀疏哈希值 sparse_hash
    idx_query = F.sphashquery(pc_hash, sparse_hash) #通过 sphashquery 函数，将原始点云哈希值 pc_hash 映射到稀疏哈希值 sparse_hash，返回索引 idx_query
    counts = F.spcount(idx_query.int(), len(sparse_hash)) #统计每个体素内的点数

    inserted_coords = F.spvoxelize(torch.floor(new_float_coord), idx_query,
                                   counts) #spvoxelize 用于对输入点云特征和坐标进行体素化处理
    inserted_coords = torch.round(inserted_coords).int() #inserted_coords 是最终的整数体素坐标
    inserted_feat = F.spvoxelize(z.F, idx_query, counts) #inserted_feat 是插值后的体素特征

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
#将点云数据从 PointTensor 转换为目标体素表示 SparseTensor
def point_to_voxel(x, z): #x: 输入的目标体素张量，类型为 SparseTensor z: 输入的点云数据，类型为 PointTensor
    if z.additional_features is None or z.additional_features.get('idx_query') is None\
       or z.additional_features['idx_query'].get(x.s) is None:
        #pc_hash = hash_gpu(torch.floor(z.C).int())
        pc_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F.sphash(x.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False): #将体素特征 SparseTensor 重新映射到点云空间 PointTensor
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        #old_hash = kernel_hash_gpu(torch.floor(z.C).int(), off)
        old_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = F.sphash(x.C.to(z.F.device))
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = F.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s),
                                  z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor
