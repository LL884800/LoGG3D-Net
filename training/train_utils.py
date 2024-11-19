import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from loss.global_loss import *
from loss.local_consistency_loss import *
#该函数根据配置中的 cfg.train_loss_function 字段返回相应的损失函数
def get_loss_function(cfg):
    if cfg.train_loss_function == 'triplet': #如果配置中指定 train_loss_function 为 'triplet'，则使用 triplet_loss（三元组损失函数）
        loss_function = triplet_loss
    elif cfg.train_loss_function == 'quadruplet': #如果指定为 'quadruplet'，则使用 quadruplet_loss（四元组损失函数）
        loss_function = quadruplet_loss 
    else:
        raise NotImplementedError(cfg.train_loss_function) #如果指定的损失函数不在支持的范围内，则抛出 NotImplementedError
    return loss_function

def get_point_loss_function(cfg):
    if cfg.point_loss_function == 'contrastive': #如果 point_loss_function 是 'contrastive'，使用 point_contrastive_loss（对比损失）
        point_loss_function = point_contrastive_loss
    elif cfg.point_loss_function == 'infonce': #如果是 'infonce'，则使用 point_infonce_loss（InfoNCE 损失）
        point_loss_function = point_infonce_loss
    else:
        raise NotImplementedError(cfg.point_loss_function) #不支持的损失函数类型会抛出 NotImplementedError
    return point_loss_function   
#get_optimizer(cfg, model_parameters) 根据配置返回优化器，用于更新模型的参数
def get_optimizer(cfg, model_parameters):
    if cfg.optimizer == 'sgd': #sgd'：随机梯度下降法（SGD），使用动量（momentum）
        optimizer = torch.optim.SGD(
            model_parameters, cfg.base_learning_rate, momentum=cfg.momentum)
    elif cfg.optimizer == 'adam': #'adam'：自适应优化算法（Adam）
        optimizer = torch.optim.Adam(model_parameters, cfg.base_learning_rate)  
    else:
        raise NotImplementedError(cfg.optimizer)
    return optimizer 

def get_scheduler(cfg, optimizer): #返回学习率调度器，用于控制学习率的变化
    # See: https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling
    if cfg.scheduler == 'lambda': #恒定学习率
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1)
    elif cfg.scheduler == 'cosine': #余弦退火调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10)
    elif cfg.scheduler == 'multistep': #多步调度器
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40], gamma=0.1)
    else:
        raise NotImplementedError(cfg.scheduler)
    return scheduler
