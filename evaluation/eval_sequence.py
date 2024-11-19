from scipy.spatial.distance import cdist
import logging
import matplotlib.pyplot as plt
import pickle
import os
import sys
import numpy as np
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from models.pipelines.pipeline_utils import *
from utils.data_loaders.make_dataloader import *
from utils.misc_utils import *
from utils.data_loaders.mulran.mulran_dataset import load_poses_from_csv, load_timestamps_csv
from utils.data_loaders.kitti.kitti_dataset import load_poses_from_txt, load_timestamps

__all__ = ['evaluate_sequence_reg']


def save_pickle(data_variable, file_name): //data_variable：要保存的变量，可以是任何 Python 对象
//file_name：保存文件的路径和文件名，指定将数据保存到哪个文件
    dbfile2 = open(file_name, 'ab') //打开指定的文件（file_name）。文件模式是 'ab'，表示以追加（二进制）模式打开文件//a：表示“追加”，如果文件存在，会在文件末尾添加内容；如果文件不存在，会创建一个新文件
//b：表示以二进制模式打开文件，这对于保存非文本数据（如序列化的对象）是必要的
    pickle.dump(data_variable, dbfile2) //将 data_variable 进行序列化，并将其写入到打开的文件 dbfile2 中
    dbfile2.close()
    logging.info(f'Finished saving: {file_name}')


def evaluate_sequence_reg(model, cfg): //model：模型对象，通常是经过训练的神经网络模型，cfg：配置对象，包含了评估所需的各种配置参数
    save_descriptors = cfg.eval_save_descriptors //是否保存描述符（特征描述）
    save_counts = cfg.eval_save_counts //是否保存计数（如正负样本的计数）
    plot_pr_curve = cfg.eval_plot_pr_curve //是否绘制精确度-召回曲线（Precision-Recall Curve）
    revisit_json_file = 'is_revisit_D-{}_T-{}.json'.format(
        int(cfg.revisit_criteria), int(cfg.skip_time))
//通过格式化字符串构造一个 JSON 文件的文件名，这个文件包含了某些序列在不同时间点是否为重访位置的信息。cfg.revisit_criteria 和 cfg.skip_time 来自配置文件


//如果数据集是kitti
if 'Kitti' in cfg.eval_dataset:
        eval_seq = cfg.kitti_eval_seq //获取要评估的序列
        cfg.kitti_data_split['test'] = [eval_seq] //设置为包含当前评估序列
        eval_seq = '%02d' % eval_seq
        sequence_path = cfg.kitti_dir + 'sequences/' + eval_seq + '/'
        _, positions_database = load_poses_from_txt(
            sequence_path + 'poses.txt')  //加载位姿信息
        timestamps = load_timestamps(sequence_path + 'times.txt')  //加载时间戳
        revisit_json_dir = os.path.join(
            os.path.dirname(__file__), '../config/kitti_tuples/')
        revisit_json = json.load(
            open(revisit_json_dir + revisit_json_file, "r"))
        is_revisit_list = revisit_json[eval_seq]

//如果数据集是mulran
    elif 'MulRan' in cfg.eval_dataset:
        eval_seq = cfg.mulran_eval_seq
        cfg.mulran_data_split['test'] = [eval_seq]  //设置为包含当前评估序列
        sequence_path = cfg.mulran_dir + eval_seq
        _, positions_database = load_poses_from_csv(
            sequence_path + '/scan_poses.csv')  //位姿信息
        timestamps = load_timestamps_csv(sequence_path + '/scan_poses.csv')  //时间戳
        revisit_json_dir = os.path.join(
            os.path.dirname(__file__), '../config/mulran_tuples/')     //加载和重访json文件
        revisit_json = json.load( //标记序列中的位置是否是重访位置
            open(revisit_json_dir + revisit_json_file, "r"))
        is_revisit_list = revisit_json[eval_seq]

    logging.info(f'Evaluating sequence {eval_seq} at {sequence_path}')
    thresholds = np.linspace(
        cfg.cd_thresh_min, cfg.cd_thresh_max, int(cfg.num_thresholds))  //设置不同的阈值，用于后续评估过程

    test_loader = make_data_loader(cfg,  
                                   cfg.test_phase,
                                   cfg.eval_batch_size,
                                   num_workers=cfg.test_num_workers,
                                   shuffle=False)  //创建数据加载器

    iterator = test_loader.__iter__()
    logging.info(f'len_dataloader {len(test_loader.dataset)}')
//初始化评估所需的变量
    num_queries = len(positions_database)
    num_thresholds = len(thresholds)

    # Databases of previously visited/'seen' places.
    seen_poses, seen_descriptors, seen_feats = [], [], []
//存储之前访问过的位姿，描述符和特征

    # Store results of evaluation.
    num_true_positive = np.zeros(num_thresholds)
    num_false_positive = np.zeros(num_thresholds)
    num_true_negative = np.zeros(num_thresholds)
    num_false_negative = np.zeros(num_thresholds)
//用于存储之前访问的位姿、描述符和特征

    prep_timer, desc_timer, ret_timer = Timer(), Timer(), Timer()

    min_min_dist = 1.0
    max_min_dist = 0.0
    num_revisits = 0  //计数是否发生重访
    num_correct_loc = 0
    start_time = timestamps[0] //设置为第一个时间戳
//迭代每一个查询
    for query_idx in range(num_queries):

        input_data = next(iterator) //获取下一个数据批次
        prep_timer.tic() //用来记录数据准备的时间
        lidar_pc = input_data[0][0]  # .cpu().detach().numpy()
        if not len(lidar_pc) > 0:
            logging.info(f'Corrupt cloud id: {query_idx}')
            continue
        input = make_sparse_tensor(lidar_pc, cfg.voxel_size).cuda()
        prep_timer.toc()
        desc_timer.tic()
        output_desc, output_feats = model(input)  # .squeeze()  //output_desc 是计算得到的全局描述符，output_feats 是局部特征
        desc_timer.toc() //用来记录描述符计算的时间
        output_feats = output_feats[0]
        global_descriptor = output_desc.cpu().detach().numpy()//.cpu().detach().numpy() 将结果从 GPU 转移到 CPU，并转换为 NumPy 数组

        global_descriptor = np.reshape(global_descriptor, (1, -1)) //global_descriptor 被重塑为一行，便于后续处理
        query_pose = positions_database[query_idx] //query_pose 获取当前查询的位姿
        query_time = timestamps[query_idx] //query_time 获取当前查询的时间戳

        if len(global_descriptor) < 1:
            continue

        seen_descriptors.append(global_descriptor)
        seen_poses.append(query_pose)

        if (query_time - start_time - cfg.skip_time) < 0:
            continue  //如果当前查询时间减去 start_time 和 skip_time 小于 0，则跳过该查询

        if save_descriptors:
            feats = output_feats.cpu().detach().numpy()
            seen_feats.append(feats)
            continue

        # Build retrieval database using entries 30s prior to current query.
        tt = next(x[0] for x in enumerate(timestamps)
                  if x[1] > (query_time - cfg.skip_time))
        db_seen_descriptors = np.copy(seen_descriptors)
        db_seen_poses = np.copy(seen_poses)
        db_seen_poses = db_seen_poses[:tt+1]
        db_seen_descriptors = db_seen_descriptors[:tt+1]
        db_seen_descriptors = db_seen_descriptors.reshape(
            -1, np.shape(global_descriptor)[1])
        //使用 timestamps 找到离当前查询时间最接近且在 skip_time 时间之前的时间点（即 tt）。截取到 tt+1 为止的所有描述符和位姿，构建一个检索数据库

        # Find top-1 candidate.
        nearest_idx = 0
        min_dist = math.inf
//用于存储最小距离和最相似的候选索引
        ret_timer.tic()
        feat_dists = cdist(global_descriptor, db_seen_descriptors,
                           metric=cfg.eval_feature_distance).reshape(-1)
        min_dist, nearest_idx = np.min(feat_dists), np.argmin(feat_dists)
        ret_timer.toc()

        place_candidate = seen_poses[nearest_idx]
        p_dist = np.linalg.norm(query_pose - place_candidate)
//place_candidate 是距离最近的候选位姿，p_dist 是查询位姿和候选位姿之间的欧几里得距离

        # is_revisit = check_if_revisit(query_pose, db_seen_poses, cfg.revisit_criteria)
        is_revisit = is_revisit_list[query_idx]
//is_revisit 从 is_revisit_list 中获取当前查询是否为重访位置
        is_correct_loc = 0  //果是重访，则增加 num_revisits 计数
        if is_revisit:
            num_revisits += 1
            if p_dist <= cfg.revisit_criteria:
                num_correct_loc += 1
                is_correct_loc = 1

        logging.info(
            f'id: {query_idx} n_id: {nearest_idx} is_rev: {is_revisit} is_correct_loc: {is_correct_loc} min_dist: {min_dist} p_dist: {p_dist}')

        if min_dist < min_min_dist:
            min_min_dist = min_dist
        if min_dist > max_min_dist:
            max_min_dist = min_dist

        # Evaluate top-1 candidate.
        for thres_idx in range(num_thresholds):
            threshold = thresholds[thres_idx]

            if(min_dist < threshold):  # Positive Prediction
                if p_dist <= cfg.revisit_criteria:
                    num_true_positive[thres_idx] += 1

                elif p_dist > cfg.not_revisit_criteria:
                    num_false_positive[thres_idx] += 1

            else:  # Negative Prediction
                if(is_revisit == 0):
                    num_true_negative[thres_idx] += 1
                else:
                    num_false_negative[thres_idx] += 1

    F1max = 0.0
    Precisions, Recalls = [], []
    if not save_descriptors:
        for ithThres in range(num_thresholds):
            nTrueNegative = num_true_negative[ithThres]
            nFalsePositive = num_false_positive[ithThres]
            nTruePositive = num_true_positive[ithThres]
            nFalseNegative = num_false_negative[ithThres]

            Precision = 0.0
            Recall = 0.0
            F1 = 0.0

            if nTruePositive > 0.0:
                Precision = nTruePositive / (nTruePositive + nFalsePositive)
                Recall = nTruePositive / (nTruePositive + nFalseNegative)

                F1 = 2 * Precision * Recall * (1/(Precision + Recall))

            if F1 > F1max:
                F1max = F1
                F1_TN = nTrueNegative
                F1_FP = nFalsePositive
                F1_TP = nTruePositive
                F1_FN = nFalseNegative
                F1_thresh_id = ithThres
            Precisions.append(Precision)
            Recalls.append(Recall)
        logging.info(f'num_revisits: {num_revisits}')
        logging.info(f'num_correct_loc: {num_correct_loc}')
        logging.info(
            f'percentage_correct_loc: {num_correct_loc*100.0/num_revisits}')
        logging.info(
            f'min_min_dist: {min_min_dist} max_min_dist: {max_min_dist}')
        logging.info(
            f'F1_TN: {F1_TN} F1_FP: {F1_FP} F1_TP: {F1_TP} F1_FN: {F1_FN}')
        logging.info(f'F1_thresh_id: {F1_thresh_id}')
        logging.info(f'F1max: {F1max}')

        if plot_pr_curve:
            plt.title('Seq: ' + str(eval_seq) +
                      '    F1Max: ' + "%.4f" % (F1max))
            plt.plot(Recalls, Precisions, marker='.')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.axis([0, 1, 0, 1.1])
            plt.xticks(np.arange(0, 1.01, step=0.1))
            plt.grid(True)
            save_dir = os.path.join(os.path.dirname(__file__), 'pr_curves')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            eval_seq = str(eval_seq).split('/')[-1]
            plt.savefig(save_dir + '/' + eval_seq + '.png')

    if not save_descriptors:
        logging.info('Average times per scan:')
        logging.info(
            f"--- Prep: {prep_timer.avg}s Desc: {desc_timer.avg}s Ret: {ret_timer.avg}s ---")
        logging.info('Average total time per scan:')
        logging.info(
            f"--- {prep_timer.avg + desc_timer.avg + ret_timer.avg}s ---")

    if save_descriptors:
        save_dir = os.path.join(os.path.dirname(__file__), str(eval_seq))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        desc_file_name = '/logg3d_descriptor.pickle'
        save_pickle(seen_descriptors, save_dir + desc_file_name)
        feat_file_name = '/logg3d_feats.pickle'
        save_pickle(seen_feats, save_dir + feat_file_name)

    if save_counts:
        save_dir = os.path.join(os.path.dirname(
            __file__), 'pickles/', str(eval_seq))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_pickle(num_true_positive, save_dir + '/num_true_positive.pickle')
        save_pickle(num_false_positive, save_dir +
                    '/num_false_positive.pickle')
        save_pickle(num_true_negative, save_dir + '/num_true_negative.pickle')
        save_pickle(num_false_negative, save_dir +
                    '/num_false_negative.pickle')

    return F1max
