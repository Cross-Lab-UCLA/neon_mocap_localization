import numpy as np


def fit_plane(centers, orient_towards=None):
    centroid = np.mean(centers, axis=1, keepdims=True)
    centered = centers - centroid

    # SVD on 3xN centered points
    try:
        U, _, _ = np.linalg.svd(centered)
    except Exception:
        # sometimes SVD does not converge on poor frames or frames
        # without apriltags, so just skip those few instances
        return None, None

    normal = U[:, 2]
    normal /= np.linalg.norm(normal)

    if orient_towards is not None:
        orient_towards = np.asarray(orient_towards, dtype=float)

        # if orient_towards.ndim == 1:
        # orient_towards = orient_towards[:, np.newaxis]

        ref_vec = orient_towards - centroid.squeeze()
        print(ref_vec)

        if np.dot(normal, ref_vec) > 0:
            normal = -normal

        U[:, 2] = normal

    U[:, 0] /= np.linalg.norm(U[:, 0])
    U[:, 1] /= np.linalg.norm(U[:, 1])
    U[:, 2] /= np.linalg.norm(U[:, 2])

    return centroid, U


def fit_plane_simple(centers, orient_towards=None, from_poses=False):
    centroid = np.mean(centers, axis=1, keepdims=True)
    centered = centers - centroid

    x_axis = centered[:, 1] - centered[:, 0]
    y_axis = centered[:, 3] - centered[:, 0]
    normal = np.cross(centered[:, 1] - centered[:, 0], centered[:, 3] - centered[:, 0])

    if orient_towards is not None:
        orient_towards = np.asarray(orient_towards, dtype=float)

        # if orient_towards.ndim == 1:
        # orient_towards = orient_towards[:, np.newaxis]

        ref_vec = orient_towards - centroid.squeeze()

        if np.dot(normal, ref_vec) > 0:
            normal = -normal

    R = np.zeros((3, 3))
    R[:, 0] = x_axis
    R[:, 1] = y_axis
    R[:, 2] = normal

    R[:, 0] /= np.linalg.norm(R[:, 0])
    R[:, 1] /= np.linalg.norm(R[:, 1])
    R[:, 2] /= np.linalg.norm(R[:, 2])

    return centroid, R


def get_plane_coordinate_system(inlier_points):
    centroid = np.mean(inlier_points, axis=0)
    centered_points = inlier_points - centroid

    # 3. Compute SVD
    # u: Unitary arrays
    # s: Singular values (variance magnitude)
    # vh: Unitary arrays (The rows of vh are the eigenvectors/principal axes)
    u, s, vh = np.linalg.svd(centered_points)

    local_x = vh[0]
    local_y = vh[1]
    local_z = vh[2]

    return local_x, local_y, local_z
