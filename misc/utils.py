import os
import numpy as np
import open3d as o3d
import numba as nb

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

@nb.jit(nopython=True)
def curl_velodyne_data(velo_in: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    pt_num = velo_in.shape[0]
    velo_out = np.zeros_like(velo_in)

    for i in range(pt_num):
        vx, vy, vz = velo_in[i, :3]

        s = 0.5 * np.arctan2(vy, vx) / np.pi

        rx, ry, rz = s * r
        tx, ty, tz = s * t

        theta = np.sqrt(rx * rx + ry * ry + rz * rz)

        if theta > 1e-10:
            kx, ky, kz = rx / theta, ry / theta, rz / theta

            ct = np.cos(theta)
            st = np.sin(theta)

            kv = kx * vx + ky * vy + kz * vz

            velo_out[i, 0] = vx * ct + (ky * vz - kz * vy) * st + kx * kv * (1 - ct) + tx
            velo_out[i, 1] = vy * ct + (kz * vx - kx * vz) * st + ky * kv * (1 - ct) + ty
            velo_out[i, 2] = vz * ct + (kx * vy - ky * vx) * st + kz * kv * (1 - ct) + tz

        else:
            velo_out[i, :3] = vx + tx, vy + ty, vz + tz

        # intensity
        velo_out[i, 3] = velo_in[i, 3]

    return velo_out


def Rodrigues(matrix):
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """

    # Axes.
    axis = np.zeros(3, np.float64)
    axis[0] = matrix[2, 1] - matrix[1, 2]
    axis[1] = matrix[0, 2] - matrix[2, 0]
    axis[2] = matrix[1, 0] - matrix[0, 1]

    # Angle.
    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
    t = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    theta = np.arctan2(r, t - 1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return axis * theta


def remove_ground(pc: np.ndarray):
    """
    Remove ground points from point cloud N*3
    """
    N_SCAN = 64
    Horizon_SCAN = 1800
    ang_res_x = 360.0 / Horizon_SCAN
    ang_res_y = 26.9 / (N_SCAN - 1)
    ang_bottom = 25.0
    sensor_mount_angle = 0.0  # Assuming sensor is mounted horizontally.

    ground_mat = np.zeros((N_SCAN, Horizon_SCAN))
    ground_mat.fill(-1)  # Initial value for all pixels is -1 (no info).

    range_image = np.zeros((N_SCAN, Horizon_SCAN, pc.shape[1]))
    for i in range(pc.shape[0]):
        vertical_angle = np.arctan2(pc[i, 2], np.sqrt(pc[i, 0] ** 2 + pc[i, 1] ** 2)) * 180 / np.pi
        row_id = (vertical_angle + ang_bottom) / ang_res_y
        if row_id < 0 or row_id >= N_SCAN:
            continue
        horizon_angle = np.arctan2(pc[i, 0], pc[i, 1]) * 180 / np.pi
        column_id = -np.round((horizon_angle - 90.0) / ang_res_x) + Horizon_SCAN / 2
        if column_id >= Horizon_SCAN:
            column_id -= Horizon_SCAN
        if column_id < 0 or column_id >= Horizon_SCAN:
            continue
        range_pc = np.sqrt(pc[i, 0] ** 2 + pc[i, 1] ** 2 + pc[i, 2] ** 2)
        if range_pc < 2:
            continue
        range_image[int(row_id), int(column_id), :] = pc[i, :]

    for j in range(Horizon_SCAN):
        for i in range(1, N_SCAN):
            if np.any(range_image[i - 1, j, :] == 0) or np.any(range_image[i, j, :] == 0):
                continue  # No info to check, invalid points.

            diff = range_image[i, j, :] - range_image[i - 1, j, :]
            angle = np.arctan2(diff[2], np.sqrt(diff[0] ** 2 + diff[1] ** 2)) * 180 / np.pi

            if np.abs(angle - sensor_mount_angle) <= 10:
                ground_mat[i - 1, j] = 1
                ground_mat[i, j] = 1

    # Extract ground cloud (groundMat == 1).
    ground_cloud = []
    for i in range(N_SCAN):
        for j in range(Horizon_SCAN):
            if ground_mat[i, j] == 1:
                ground_cloud.append(range_image[i, j, :])

    # Apply the mask to the range image to get the non-ground points.
    non_ground_cloud = range_image[ground_mat[:, :] != 1]

    return non_ground_cloud


def create_point_cloud(points, color=[0, 0.651, 0.929]):
    # 1.000, 0.706, 0.000
    xyz = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color(color)
    return pcd


def draw_pairs(src, dst, tf, window_name="pairs"):
    src = create_point_cloud(src, color=[0, 0.651, 0.929])
    dst = create_point_cloud(dst, color=[1.000, 0.706, 0.000])
    src = src.transform(tf)
    o3d.visualization.draw_geometries([src, dst], window_name=window_name)


def draw_pc(pc, window_name="pc"):
    pc = create_point_cloud(pc, color=[0, 0.651, 0.929])
    o3d.visualization.draw_geometries([pc], window_name=window_name)
