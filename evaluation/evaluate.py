import os  //路径操作
import sys //系统交互
import torch  //深度学习模型的加载和运行
import logging  //输出日志
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))  //将脚本所在目录的父目录添加到sys.path,使得脚本能够访问父目录中的模块或文件
from utils.misc_utils import log_config
from evaluation.eval_sequence import *

ch = logging.StreamHandler(sys.stdout)  //设置日志输出到标准输出（命令行）
logging.getLogger().setLevel(logging.INFO)   //设置日志的最低级别时info，就是显示所有info及以上的日志信息
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")  //用于设置配置日志的输出格式和时间格式


def evaluate_checkpoint(model, save_path, cfg):
  //用于加载模型的检查点，恢复模型的状态并进行评估
    checkpoint = torch.load(save_path)  # ,map_location='cuda:0')  //从指定路径加载模型检查点
    model.load_state_dict(checkpoint['model_state_dict']) //将保存的模型参数加载到模型中

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model = model.cuda()
    model.eval()

    return evaluate_sequence_reg(model, cfg)


if __name__ == "__main__":
    from models.pipeline_factory import get_pipeline
    from config.eval_config import get_config_eval

    cfg = get_config_eval() //获取评估配置对象

    # Get model
    model = get_pipeline(cfg.eval_pipeline)

    save_path = os.path.join(os.path.dirname(__file__), '../', 'checkpoints')
    save_path = str(save_path) + cfg.checkpoint_name
    print('Loading checkpoint from: ', save_path)
    logging.info('\n' + ' '.join([sys.executable] + sys.argv)) //记录当前运行的 Python 程序和命令行参数，方便调试
    log_config(cfg, logging)

    eval_F1_max = evaluate_checkpoint(model, save_path, cfg)
    logging.info(
        '\n' + '******************* Evaluation Complete *******************')
    logging.info('Checkpoint Name: ' + str(cfg.checkpoint_name))
    if 'Kitti' in cfg.eval_dataset:
        logging.info('Evaluated Sequence: ' + str(cfg.kitti_eval_seq))
    elif 'MulRan' in cfg.eval_dataset:
        logging.info('Evaluated Sequence: ' + str(cfg.mulran_eval_seq))
    logging.info('F1 Max: ' + str(eval_F1_max))
