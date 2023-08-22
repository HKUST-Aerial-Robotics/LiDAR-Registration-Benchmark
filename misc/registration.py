import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import teaserpp_python
from misc.utils import create_point_cloud
import copy
from misc.config import cfg

def fpfh_teaser(source: np.ndarray, target: np.ndarray, visualize=False):
    source = create_point_cloud(source, color=[0, 0.651, 0.929])
    target = create_point_cloud(target, color=[1.000, 0.706, 0.000])

    VOXEL_SIZE = 0.5
    source = source.voxel_down_sample(voxel_size=cfg.fpfh.voxel_size)
    target = target.voxel_down_sample(voxel_size=cfg.fpfh.voxel_size)

    A_xyz = pcd2xyz(source)  # np array of size 3 by N
    B_xyz = pcd2xyz(target)  # np array of size 3 by M

    # extract FPFH features
    A_feats = extract_fpfh(source, cfg.fpfh.radius_normal, cfg.fpfh.radius_feature)
    B_feats = extract_fpfh(target, cfg.fpfh.radius_normal, cfg.fpfh.radius_feature)

    # establish correspondences by nearest neighbour search in feature space
    corrs_A, corrs_B = find_correspondences(A_feats, B_feats, mutual_filter=True)
    A_corr = A_xyz[:, corrs_A]  # np array of size 3 by num_corrs
    B_corr = B_xyz[:, corrs_B]  # np array of size 3 by num_corrs

    teaser_solver = get_teaser_solver(cfg.teaser.noise_bound)
    teaser_solver.solve(A_corr, B_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser, t_teaser)

    if visualize:
        A_pcd_T_teaser = copy.deepcopy(source).transform(T_teaser)
        o3d.visualization.draw_geometries([A_pcd_T_teaser, target], window_name="Registration by Teaser++")

    return T_teaser


def pcd2xyz(pcd):
    return np.asarray(pcd.points).T


def extract_fpfh(pcd, radius_normal, radius_feature):
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return np.array(fpfh.data).T


def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn, workers=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds


def find_correspondences(feats0, feats1, mutual_filter=True):
    nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
    corres10_idx1 = np.arange(len(nns10))
    corres10_idx0 = nns10

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1


def get_teaser_solver(noise_bound):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = \
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver


def Rt2T(R, t):
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def icp_registration(source, target, init_tf, voxel_size=0.3):
    # point to plane
    source_pcd = create_point_cloud(source)
    target_pcd = create_point_cloud(target)
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    loss = o3d.pipelines.registration.HuberLoss(k=0.5)
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_down, target_down, 1,
        init_tf,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    return reg_p2l.transformation
