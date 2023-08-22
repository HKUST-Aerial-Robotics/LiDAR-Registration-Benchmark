import numpy as np
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from misc.config import cfg_from_yaml_file, cfg

from misc.utils import draw_pairs
from misc.registration import icp_registration
from misc.datasets.kitti_utils import KittiDataset
from misc.datasets.kitti360_utils import Kitti360Dataset
from misc.datasets.apollo_utils import ApolloDataset
from glob import glob

def visualize_kitti360():
    kitti360 = Kitti360Dataset(cfg.kitti360_root)
    seq = 6
    i = 5248
    j = 5874

    source_cloud = kitti360.get_lidar_pc(seq, i)
    target_cloud = kitti360.get_lidar_pc(seq, j)
    tf = np.linalg.inv(kitti360.get_lidar_pose(seq, j)) @ kitti360.get_lidar_pose(seq, i)
    print(tf)
    # draw_pairs(source_cloud, target_cloud, tf, window_name="seq %d, %d -> %d" % (seq, i, j))
    tf = icp_registration(source_cloud, target_cloud, tf)
    print(tf)
    draw_pairs(source_cloud, target_cloud, tf, window_name="seq %d, %d -> %d" % (seq, i, j))

def visualize_apollo():
    apollo_dataset = ApolloDataset(cfg.apollo_root)
    seq = 20
    i = 799
    j = 3286

    source_cloud = apollo_dataset.get_lidar_pc(seq, i)
    target_cloud = apollo_dataset.get_lidar_pc(seq, j)
    tf = np.linalg.inv(apollo_dataset.get_lidar_pose(seq, j)) @ apollo_dataset.get_lidar_pose(seq, i)
    print(tf)
    draw_pairs(source_cloud, target_cloud, tf, window_name="seq %d, %d -> %d" % (seq, i, j))
    tf = icp_registration(source_cloud, target_cloud, tf)
    print(tf)
    draw_pairs(source_cloud, target_cloud, tf, window_name="seq %d, %d -> %d" % (seq, i, j))

def visualize_kitti():
    all_pairs = {}
    for file in glob(os.path.join(cfg.ROOT_DIR, "benchmarks/kitti_lc/*.txt")):
        with open(file, 'r') as f:
            data = f.readlines()[1:]
        data = [x.strip().split(' ') for x in data]
        data = np.array(data).astype(np.float32)
        seq = data[:, 0].astype(np.int32)
        i = data[:, 1].astype(np.int32)
        seq_db = data[:, 2].astype(np.int32)
        j = data[:, 3].astype(np.int32)
        for k in range(len(seq)):
            idx = (seq[k], i[k], seq_db[k], j[k])
            all_pairs[idx] = data[k, 4:20].reshape(4, 4)

    cfg.poses_root = os.path.join(cfg.ROOT_DIR, "benchmarks/sem_kitti_poses")
    kitti = KittiDataset(cfg.kitti_root, cfg.poses_root)
    for idx, tf in all_pairs.items():
        seq, i, seq_db, j = idx
        source_cloud = kitti.get_lidar_pc(seq, i)
        target_cloud = kitti.get_lidar_pc(seq_db, j)
        dist = np.linalg.norm(tf[:3, 3])
        draw_pairs(source_cloud, target_cloud, tf, window_name="seq %d, %d -> %d, dist %.2f" % (seq, i, j, dist))
        tf = icp_registration(source_cloud, target_cloud, tf)
        draw_pairs(source_cloud, target_cloud, tf, window_name="seq %d, %d -> %d, dist %.2f" % (seq, i, j, dist))

if __name__ == '__main__':

    cfg_from_yaml_file(os.path.join(cfg.ROOT_DIR, "configs/dataset.yaml"), cfg)
    # visualize_kitti360()
    # visualize_apollo()
    visualize_kitti()