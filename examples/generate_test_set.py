import numpy as np
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from misc.config import cfg_from_yaml_file, cfg

from misc.datasets.kitti360_utils import Kitti360Dataset
from misc.datasets.apollo_utils import ApolloDataset
from misc.datasets.kitti_utils import KittiDataset
from misc.utils import mkdir
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/dataset.yaml')
    parser.add_argument('--dataset', type=str, default='kitti_lc')
    args = parser.parse_args()
    cfg_from_yaml_file(os.path.join(cfg.ROOT_DIR, args.config), cfg)

    cfg.SAVE_DIR = os.path.join(cfg.ROOT_DIR, "benchmarks/{}".format(args.dataset))
    mkdir(cfg.SAVE_DIR)
    print('SAVE_DIR: {}'.format(cfg.SAVE_DIR))

    if args.dataset == 'kitti360_lc':
        kitti360 = Kitti360Dataset(cfg.kitti360_root)
        kitti360.make_lc_benchmarks()
    elif args.dataset == 'apollo_lc':
        apollo = ApolloDataset(cfg.apollo_root)
        apollo.make_lc_benchmarks()
    elif args.dataset == 'kitti_lc':
        kitti = KittiDataset(cfg.kitti_root)
        kitti.make_lc_benchmarks()
    elif args.dataset == 'kitti_10m':
        kitti = KittiDataset(cfg.kitti_root)
        kitti.make_10m_benchmarks()
    else:
        raise NotImplementedError