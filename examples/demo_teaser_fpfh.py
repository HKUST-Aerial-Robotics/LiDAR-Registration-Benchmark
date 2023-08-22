import open3d as o3d
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from misc.config import cfg_from_yaml_file, cfg, Logger
from misc.registration import fpfh_teaser
import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='./benchmarks/kitti_10m/test.txt')
    parser.add_argument('--config', type=str, default='./configs/dataset.yaml')
    args = parser.parse_args()
    cfg_from_yaml_file(os.path.join(cfg.ROOT_DIR, args.config), cfg)

    logger = Logger("demo_teaser_fpfh")
    pcd_path_s = os.path.join(cfg.ROOT_DIR, 'docs/pcds', 'apollo_21_36822.pcd')
    pcd_path_t = os.path.join(cfg.ROOT_DIR, 'docs/pcds', 'apollo_21_38908.pcd')
    pcd_s = o3d.io.read_point_cloud(pcd_path_s)
    pcd_t = o3d.io.read_point_cloud(pcd_path_t)

    points_s = np.asarray(pcd_s.points)
    points_t = np.asarray(pcd_t.points)
    T = fpfh_teaser(points_s, points_t, False)

    # visualize
    pcd_s.paint_uniform_color([0, 0.651, 0.929])
    pcd_t.paint_uniform_color([1.000, 0.706, 0.000])
    o3d.visualization.draw_geometries([pcd_s, pcd_t], window_name="Before registration")

    pcd_s.transform(T)
    o3d.visualization.draw_geometries([pcd_s, pcd_t], window_name="After registration")
