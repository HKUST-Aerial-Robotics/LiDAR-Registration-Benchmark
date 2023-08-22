import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from misc.config import cfg
from misc.registration import icp_registration
from sklearn.neighbors import KDTree
from misc.utils import draw_pc

levels = {
    '0_10': [0, 10],
    '10_20': [10, 20],
    '20_30': [20, 30],
}

class KittiDataset:
    def __init__(self, kitti_root, poses_root=None):
        self.kitti_root = kitti_root
        self.poses_root = os.path.join(cfg.ROOT_DIR,
                                       "benchmarks/sem_kitti_poses") if poses_root is None else poses_root
        self.seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.lidar_to_cam = {}
        for seq in self.seqs:
            self.lidar_to_cam[seq] = self.get_lidar_to_cam(seq)
        self.seq_poses = {}
        self.session_trajecotries = {}
        for seq in self.seqs:
            pose_file = os.path.join(self.poses_root, "%02d.txt" % seq)
            pose_array = np.genfromtxt(pose_file).reshape((-1, 3, 4))
            self.session_trajecotries[seq] = pose_array[:, :3, 3]
            self.seq_poses[seq] = {}
            for i in range(pose_array.shape[0]):
                tf = np.eye(4)
                tf[:3, :3] = pose_array[i][:3, :3]
                tf[:3, 3] = pose_array[i][:3, 3]
                self.seq_poses[seq][i] = tf @ self.lidar_to_cam[seq]
        print("Kitti dataset loaded.")

    def make_lc_benchmarks(self):
        for level in levels.keys():
            self.make_lc_benchmark(level)

    def make_lc_benchmark(self, level):
        print("level: ", level)
        subset_names = [0, 2, 5, 6, 8]
        sample_interval = cfg.sample_interval  # sample one pose every sample_interval meters

        pairs = []
        dist_list = []
        for seq in subset_names:
            imu_poses = self.session_trajecotries[seq]
            trans_range = levels[level]
            kdtree = KDTree(imu_poses)
            # draw_pc(imu_poses, "kitti_%s" % seq); continue
            for i in tqdm(range(imu_poses.shape[0]), desc="seq %d" % seq):
                # Query the KDTree for points within the trans_range
                ind = kdtree.query_radius(imu_poses[i].reshape(1, -1), r=trans_range[1])

                for j in ind[0]:
                    j = int(j)
                    if j <= i or j - i <= cfg.skip_frames:
                        # Skip if the same point, or the points are not far enough apart
                        continue

                    lidar_pose_i = self.get_lidar_pose(seq, i)
                    lidar_pose_j = self.get_lidar_pose(seq, j)

                    # Compute the Euclidean distance
                    dist = np.linalg.norm((lidar_pose_i[:3, 3] - lidar_pose_j[:3, 3]))
                    # Skip if the distance is not within the trans_range
                    if dist <= trans_range[0] or dist >= trans_range[1]:
                        continue

                    # Check the sample_interval condition with the last added pair
                    if len(pairs) > 0 and pairs[-1][0] == seq:
                        dist_i = np.linalg.norm(
                            (lidar_pose_i[:3, 3] - self.get_lidar_pose(seq, pairs[-1][1])[:3, 3]))
                        dist_j = np.linalg.norm(
                            (lidar_pose_j[:3, 3] - self.get_lidar_pose(seq, pairs[-1][2])[:3, 3]))
                        if dist_i < sample_interval or dist_j < sample_interval:
                            continue

                    # Add the pair and the distance to the lists
                    pairs.append((seq, i, j))
                    dist_list.append(dist)

        print("total pairs: ", len(pairs))
        pairs = self.refine_poses(pairs, refine=True)

        num_test = len(pairs)
        save_path = os.path.abspath(os.path.join(cfg.SAVE_DIR, "test_%s.txt" % level))
        with open(save_path, "w") as f:
            f.write("seq i seq_db j mot1 mot2 mot3 mot4 mot5 mot6 mot7 mot8 mot9 mot10 mot11 mot12 mot13 mot14 mot15\n")
            for p in pairs:
                f.write("%d %d %d %d" % (p[0], p[1], p[0], p[2]))
                for i in range(3, len(p)):
                    f.write(" %f" % p[i])
                f.write("\n")

    def make_10m_benchmarks(self):
        # Refer to Fully convolutional geometric features, ICCV 2019
        subset_names = [8, 9, 10]
        pairs = []
        for drive_id in subset_names:
            positions = self.session_trajecotries[drive_id]
            inames = np.arange(len(positions)).tolist()

            pdist = (positions.reshape(1, -1, 3) - positions.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1))
            more_than_10 = pdist > 10
            curr_time = inames[0]
            while curr_time in inames:
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                    continue
                else:
                    next_time = (next_time[0] + curr_time - 1).item()

                if inames.count(curr_time) > 0 and inames.count(next_time) > 0:
                    pairs.append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1

        print("start refine_poses")
        pairs = self.refine_poses(pairs)

        num_test = len(pairs)
        save_path = os.path.abspath(os.path.join(cfg.SAVE_DIR, "test.txt"))
        print("Num_test", num_test, " ,saving to", save_path)
        with open(save_path, "w") as f:
            f.write("seq i seq_db j mot1 mot2 mot3 mot4 mot5 mot6 mot7 mot8 mot9 mot10 mot11 mot12 mot13 mot14 mot15\n")
            for p in pairs:
                f.write("%d %d %d %d" % (p[0], p[1], p[0], p[2]))
                for i in range(3, len(p)):
                    f.write(" %f" % p[i])
                f.write("\n")

    def get_lidar_pose(self, seq, frame_id):
        return self.seq_poses[seq][frame_id]

    def get_lidar_pc(self, seq, frame_id):
        pc_file = os.path.join(self.kitti_root, "sequences/%02d/velodyne/%06d.bin" % (seq, frame_id))
        pc = np.fromfile(pc_file, dtype=np.float32).reshape((-1, 4))
        return pc

    def get_lidar_to_cam(self, seq):
        calib_file = os.path.join(self.kitti_root, "sequences/%02d" % seq, "calib.txt")
        with open(calib_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("Tr:"):
                    line = line.split(" ")[1:]
                    tf = np.array(line, dtype=np.float32).reshape((3, 4))
                    tf = np.vstack((tf, np.array([0, 0, 0, 1], dtype=np.float32)))
                    return tf

    def get_tf(self, seq, idx_1, idx_2):
        # from idx_1 to idx_2
        pose1 = self.get_lidar_pose(seq, idx_1)
        pose2 = self.get_lidar_pose(seq, idx_2)
        tf = np.eye(4)
        tf[:3, :3] = pose2[:3, :3].T.dot(pose1[:3, :3])
        tf[:3, 3] = pose2[:3, :3].T @ (pose1[:3, 3] - pose2[:3, 3])
        return tf

    def refine_poses(self, pairs, refine=True):
        refined_pairs = []
        for seq, i, j in tqdm(pairs):
            tf = self.get_tf(seq, i, j)
            # draw_pairs(source_cloud, target_cloud, tf, window_name="before")
            if refine:
                source_cloud = self.get_lidar_pc(seq, i)
                target_cloud = self.get_lidar_pc(seq, j)
                tf = icp_registration(source_cloud, target_cloud, tf)
            # draw_pairs(source_cloud, target_cloud, tf, window_name="after")
            refined_pair = [seq, i, j]
            refined_pair.extend(tf.flatten())
            refined_pairs.append(refined_pair)
        return refined_pairs


def read_trajectory(poses_path, seq):
    pose_file = os.path.join(poses_path, "%02d.txt" % seq)
    pose_array = np.genfromtxt(pose_file)
    pose_array = pose_array.reshape((-1, 3, 4))
    trajectory = []
    for i in range(pose_array.shape[0]):
        tf = np.eye(4)
        tf[:3, :3] = pose_array[i][:3, :3]
        tf[:3, 3] = pose_array[i][:3, 3]
        trajectory.append(tf)
    trajectory = np.array(trajectory)
    trajectory = trajectory[:, :3, 3]
    return trajectory
