import os
import sys
sys.path.append(os.path.dirname(__file__))
from pipelines.PointNetVLAD import *
from pipelines.LOGG3D import *


def get_pipeline(pipeline_name):
    if pipeline_name == 'LOGG3D': 
        pipeline = LOGG3D(feature_dim=16) #初始化时指定特征维度为 16
    elif pipeline_name == 'PointNetVLAD':
        #是否提取全局特征 global_feat，是否应用特征变换 feature_transform，是否使用最大池化 max_pool，输出特征维度为 256，每个点云包含 4096 个点
        pipeline = PointNetVLAD(global_feat=True, feature_transform=True,
                                max_pool=False, output_dim=256, num_points=4096)
    return pipeline


if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '../')) #再次动态添加路径，将父目录加入搜索路径
    #get_config() 是从 eval_config 模块中导入的函数，用于获取训练配置
    from config.eval_config import get_config
    cfg = get_config()
    #根据配置中的 train_pipeline 参数选择对应的管道模型（如 PointNetVLAD 或 LOGG3D），并将模型加载到 GPU (cuda())
    model = get_pipeline(cfg.train_pipeline).cuda()
    # print(model)

    from utils.data_loaders.make_dataloader import *
    #使用 make_data_loader 函数加载训练数据
    train_loader = make_data_loader(cfg,
                                    cfg.train_phase, #当前训练阶段
                                    cfg.batch_size, #批量大小
                                    num_workers=cfg.train_num_workers, #数据加载线程数
                                    shuffle=True) #是否随机打乱数据
    iterator = train_loader.__iter__() #获取数据加载器的迭代器 iterator
    l = len(train_loader.dataset) #遍历整个数据集 l = len(train_loader.dataset)
    for i in range(l):
        input_batch = next(iterator)
        input_st = input_batch[0].cuda()
        output = model(input_st)
        print('')
       #对每个批次数据：提取点云输入数据 input_st，并将其加载到 GPU。调用模型处理点云数据，得到输出 output。最后输出结果（具体内容尚未实现，仅打印空行） 
