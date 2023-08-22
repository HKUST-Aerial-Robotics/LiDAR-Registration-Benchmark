import numpy as np
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from misc.config import cfg_from_yaml_file, cfg, Logger
from misc.datasets.kitti_utils import KittiDataset
from misc.datasets.kitti360_utils import Kitti360Dataset
from misc.datasets.apollo_utils import ApolloDataset
from tqdm import tqdm
from time import perf_counter
import argparse
try:
    from misc.registration import fpfh_teaser
except:
    print("Please install teaserpp-python first. See https://github.com/MIT-SPARK/TEASER-plusplus")

logger = Logger("evaluate")

def read_test_file(test_file):
    test_pairs = []
    for row, line in enumerate(open(test_file)):
        if row == 0:
            continue
        line = line.strip()
        if len(line) == 0:
            continue
        line = line.split()
        seq, i, seq_db, j = [int(x) for x in line[:4]]
        tf = np.array([float(x) for x in line[4:20]]).reshape(4, 4)
        overlap = 1.0
        if len(line) > 20:
            overlap = float(line[20])
        test_pairs.append((seq, i, seq_db, j, tf, overlap))
    return test_pairs

class EvalResult:
    def __init__(self, success=0, total=0, t: float=0):
        self.success = success
        self.total = total
        self.time = t

    def __add__(self, other):
        self.success += other.success
        self.total += other.total
        self.time += other.time
        return self

    def evaluate(self, tf, gt):
        rot_est = tf[:3, :3]
        rot_gt = gt[:3, :3]
        trace = np.trace(np.dot(rot_est, rot_gt.T))
        tmp = np.clip((trace - 1) / 2, -1, 1)
        rot_succ = np.arccos(tmp) * 180 / np.pi < cfg.evaluation.rot_thd

        trans_est = tf[:3, 3]
        trans_gt = gt[:3, 3]
        trans_succ = np.linalg.norm(trans_gt - trans_est) < cfg.evaluation.trans_thd

        return rot_succ and trans_succ

    def print(self):
        logger.info("Success rate: %.2f%%" % (self.success / self.total * 100))
        logger.info("Average time: %.2fms" % (self.time / self.total * 1000))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='./benchmarks/kitti_10m/test.txt')
    parser.add_argument('--config', type=str, default='./configs/dataset.yaml')
    args = parser.parse_args()
    cfg_from_yaml_file(os.path.join(cfg.ROOT_DIR, args.config), cfg)

    test_file = os.path.join(cfg.ROOT_DIR, args.test_file)
    dataset_name = os.path.basename(os.path.dirname(args.test_file))
    logger.info("Dataset: %s.\nTest file: %s." % (dataset_name, test_file))
    if 'kitti' in dataset_name.lower():
        dataset = KittiDataset(cfg.kitti_root)
    elif 'apollo' in dataset_name.lower():
        dataset = ApolloDataset(cfg.apollo_root)
    elif 'kitti360' in dataset_name.lower():
        dataset = Kitti360Dataset(cfg.kitti360_root)
    test_pairs = read_test_file(test_file)

    eval_results = {}
    eval_all = EvalResult()
    for seq, i, seq_db, j, tf_gt, overlap in tqdm(test_pairs):
        source_cloud = dataset.get_lidar_pc(seq, i)
        target_cloud = dataset.get_lidar_pc(seq_db, j)
        t1 = perf_counter()
        # your registration code here
        try:
            tf = fpfh_teaser(source_cloud, target_cloud, False)
        except:
            tf = np.eye(4)
        t2 = perf_counter()

        # evaluate
        if not seq in eval_results:
            eval_results[seq] = EvalResult()
        eval_results[seq] += EvalResult(success=eval_all.evaluate(tf, tf_gt), total=1, t=(t2 - t1))
        eval_all += EvalResult(success=eval_all.evaluate(tf, tf_gt), total=1, t=(t2 - t1))

    logger.info("All sequences:")
    eval_all.print()
    logger.info("Each sequence:")
    for seq in eval_results:
        logger.info("Sequence %d:" % seq)
        eval_results[seq].print()



