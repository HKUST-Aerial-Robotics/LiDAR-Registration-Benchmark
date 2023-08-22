import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from misc.config import cfg
from misc.utils import draw_pairs
from misc.registration import icp_registration
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import KDTree
from misc.utils import draw_pc

levels = {
    '0_10': [0, 10],
    '10_20': [10, 20],
    '20_30': [20, 30],
    '30_40': [30, 40],
}

class ApolloDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sessions = {
            20: "TestData/HighWay237/2018-10-12/",
            21: "TestData/SunnyvaleBigloop/2018-10-03/",
            22: "TestData/MathildaAVE/2018-10-12/",
            23: "TestData/SanJoseDowntown/2018-10-11/2/",
            24: "TestData/SanJoseDowntown/2018-10-11/1/",
            25: "TestData/BaylandsToSeafood/2018-10-12/",
            26: "TestData/ColumbiaPark/2018-10-11/",
        }
        self.session_poses = {}
        self.frame_ids = {}
        self.session_trajectories = {}
        for drive_id in self.sessions.keys():
            self.session_poses[drive_id] = {}
            pose_file = os.path.join(self.root_dir, self.sessions[drive_id], 'poses', 'gt_poses.txt')
            pose_array = np.genfromtxt(pose_file)
            self.frame_ids[drive_id] = pose_array[:, 0]
            for i in range(pose_array.shape[0]):
                tf = np.eye(4)
                tf[:3, :3] = R.from_quat(pose_array[i, 5:]).as_matrix()
                tf[:3, 3] = pose_array[i, 2:5]
                self.session_poses[drive_id][self.frame_ids[drive_id][i]] = tf
            self.session_trajectories[drive_id] = pose_array[:, 2:5]
        print("Apollo dataset loaded.")

    def make_lc_benchmarks(self):
        for level in levels.keys():
            self.make_lc_benchmark(level)

    def make_lc_benchmark(self, level):
        print("level: ", level)
        sample_interval = cfg.sample_interval  # sample one pose every sample_interval meters
        pairs = []
        dist_list = []
        apollo_dataset = ApolloDataset(cfg.apollo_root)
        for drive_id in apollo_dataset.sessions.keys():
            positions = apollo_dataset.session_trajectories[drive_id]
            trans_range = levels[level]
            # draw_pc(positions, "apollo_%s" % drive_id); continue
            # get all loop pairs using the selected map
            kdtree = KDTree(positions)
            for i in tqdm(range(len(positions)), desc=apollo_dataset.sessions[drive_id]):
                # Query the KDTree for points within the trans_range
                ind = kdtree.query_radius(positions[i].reshape(1, -1), r=trans_range[1])

                for j in ind[0]:
                    j = int(j)
                    if j <= i or j - i <= cfg.skip_frames:
                        # Skip if the same point, or the points are not far enough apart
                        continue

                    # Compute the Euclidean distance
                    dist = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))

                    # Skip if the distance is not within the trans_range
                    if dist <= trans_range[0] or dist >= trans_range[1]:
                        continue

                    # Check the sample_interval condition with the last added pair
                    if len(pairs) > 0 and pairs[-1][0] == drive_id:
                        dist_i = np.sqrt(
                            np.sum((positions[i] - apollo_dataset.get_lidar_pose(drive_id, pairs[-1][1])[:3, 3]) ** 2))
                        dist_j = np.sqrt(
                            np.sum((positions[j] - apollo_dataset.get_lidar_pose(drive_id, pairs[-1][2])[:3, 3]) ** 2))
                        if dist_i < sample_interval or dist_j < sample_interval:
                            continue

                    # Add the pair and the distance to the lists
                    pairs.append(
                        (drive_id, apollo_dataset.frame_ids[drive_id][i], apollo_dataset.frame_ids[drive_id][j]))
                    dist_list.append(dist)
            print("drive_id: ", drive_id, "pairs: ", len(pairs))

        print("start refine_poses")
        pairs = apollo_dataset.refine_poses(pairs, False)

        num_test = len(pairs)
        save_path = os.path.abspath(os.path.join(cfg.SAVE_DIR, "test_%s.txt" % level))
        print("Num_test", num_test, " ,saving to", save_path)
        with open(save_path, "w") as f:
            f.write("seq i seq_db j mot1 mot2 mot3 mot4 mot5 mot6 mot7 mot8 mot9 mot10 mot11 mot12 mot13 mot14 mot15\n")
            for p in pairs:
                f.write("%d %d %d %d" % (p[0], p[1], p[0], p[2]))
                for i in range(3, len(p)):
                    f.write(" %f" % p[i])
                f.write("\n")

    def get_lidar_pose(self, drive_id, frame_id):
        return self.session_poses[drive_id][frame_id]

    def read_transform(self, drive_id, idx_1, idx_2):
        # from idx_1 to idx_2
        pose1 = self.session_poses[drive_id][idx_1]
        pose2 = self.session_poses[drive_id][idx_2]
        return np.linalg.inv(pose2) @ pose1

    def get_lidar_pc(self, drive_id, id):
        pcd_path = os.path.join(cfg.apollo_root, self.sessions[drive_id], "pcds/%d.pcd" % id)
        points = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(points.points)
        return points

    def refine_poses(self, pairs, icp_refine=False):
        refined_pairs = []
        for drive_id, i, j in tqdm(pairs):
            tf = self.read_transform(drive_id, i, j)
            if icp_refine:
                source_cloud = self.get_lidar_pc(drive_id, i)
                target_cloud = self.get_lidar_pc(drive_id, j)
                draw_pairs(source_cloud, target_cloud, tf, window_name="before")
                tf = icp_registration(source_cloud, target_cloud, tf)
                draw_pairs(source_cloud, target_cloud, tf, window_name="after")
            refined_pair = [drive_id, i, j]
            refined_pair.extend(tf.flatten())
            refined_pairs.append(refined_pair)
        return refined_pairs
