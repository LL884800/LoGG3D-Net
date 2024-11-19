import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#SOP 是一个神经网络模块，用于对输入张量执行二阶池化（Second-Order Pooling）。其核心功能包括特征协方差矩阵计算、功率归一化（可选）、特征向量化及归一化。该模块可以作为深度学习流水线的一部分，用于处理具有高维特征的数据。

class SOP(nn.Module):
    def __init__(self, thresh=1e-8, is_vec=True, signed_sqrt=False, do_pe=True, do_fc=False, input_dim=16, is_tuple=False):
        super(SOP, self).__init__()
        self.thresh = thresh #数值稳定性的阈值，用于避免计算中的极端情况
        self.is_vec = is_vec #控制输出是否为向量化形式（默认是矩阵形式）
        self.do_pe = do_pe  #是否执行功率归一化（通过奇异值分解实现）
        self.sop_dim = input_dim * input_dim  #：输入特征的维度，默认为 16
        self.signed_sqrt = signed_sqrt  #是否应用带符号的平方根归一化
        self.do_fc = do_fc  #是否启用全连接层对特征进行降维
        self.is_tuple = is_tuple  #指示输入是否为元组形式（特定任务中可能需要）

        cs = [4096, 2048, 1024]  # redundant fc layers
        cr = self.sop_dim/cs[0]       #二阶池化的特征维度，等于 input_dim * input_dim
        cs = [int(cr * x) for x in cs]
        self.fc1 = nn.Linear(cs[0], cs[1])
        self.fc2 = nn.Linear(cs[1], cs[2])
#二阶池化 (_so_maxpool) 算输入特征的协方差矩阵，即通过特征交叉生成二阶统计量
    def _so_maxpool(self, x):
        while len(x.data.shape) < 4:
            x = torch.unsqueeze(x, 0)
        batchSize, tupleLen, nFeat, dimFeat = x.data.shape
        x = torch.reshape(x, (-1, dimFeat))
        x = torch.unsqueeze(x, -1)
        x = x.matmul(x.transpose(1, 2))

        x = torch.reshape(x, (batchSize, tupleLen, nFeat, dimFeat, dimFeat))
        x = torch.max(x, 2).values
        x = torch.reshape(x, (-1, dimFeat, dimFeat))
        if self.do_pe:
            x = x.double()
            # u_, s_, vh_ = torch.linalg.svd(x)
            # dist = torch.dist(x, u_ @ torch.diag_embed(s_) @ vh_)
            # dist_same = torch.allclose(x, u_ @ torch.diag_embed(s_) @ vh_)
            # s_alpha = torch.pow(s_, 0.5)
            # x = u_ @ torch.diag_embed(s_alpha) @ vh_

            # For pytorch versions < 1.9
            u_, s_, v_ = torch.svd(x)
            # dist = torch.dist(x, u_ @ torch.diag_embed(s_) @  v_.transpose(-2, -1))
            # dist_same = torch.allclose(x, u_ @ torch.diag_embed(s_) @  v_.transpose(-2, -1))
            s_alpha = torch.pow(s_, 0.5)
            x = u_ @ torch.diag_embed(s_alpha) @ v_.transpose(-2, -1)

        x = torch.reshape(x, (batchSize, tupleLen, dimFeat, dimFeat))
        return x  # .float()
"""
将输入的维度扩展成标准的 4 维形状：(batch_size, tuple_length, n_features, feature_dim)
计算协方差矩阵 
若启用功率归一化（do_pe=True），则对协方差矩阵执行奇异值分解（SVD），并进行矩阵变换
将归一化后的矩阵返回
"""

    def _l2norm(self, x): #对每个特征向量进行 L2 范数归一化，以确保特征的尺度一致性
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x

    def forward(self, x):
        x = self._so_maxpool(x) #输入张量经过 _so_maxpool 计算协方差矩阵

        if self.is_vec: #若 is_vec=True，将输出矩阵展平为向量形式
            x = torch.reshape(x, (x.size(0), x.size(1), -1))
        # if self.do_fc:
        #     x = F.relu(self.fc1(x.float()))
        #     x = F.relu(self.fc2(x))
        x = self._l2norm(x) #执行 L2 归一化
        return torch.squeeze(x) #返回最终特征

#模拟生成一个维度为 (44, 100, 64) 的随机输入数据（假设有 44 个片段，每个片段 100 个点，每个点有 64 维特征）。
#将数据扩展为批次形式，转换为 PyTorch 张量并移动到 GPU
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = SOP(signed_sqrt=False, do_fc=False) #初始化 SOP 模型，指定参数（如是否执行功率归一化等）
    model = model.to(device)
    model = nn.DataParallel(model) #利用 nn.DataParallel 包装模型以支持多 GPU
    segments = np.random.rand(44, 100, 64)*10
    feed_tensor = torch.from_numpy(segments).float()  # s.double() #将输入数据喂入模型
    feed_tensor = torch.unsqueeze(feed_tensor, 0) #获取经过二阶池化和归一化的输出
    feed_tensor = torch.reshape(feed_tensor, (2, -1, 100, 64))
    feed_tensor = feed_tensor.to(device)
    output = model(feed_tensor)
    print('')
