import numpy as np


def fit_plane(centers, orient_towards=None):
    centroid = np.mean(centers, axis=1, keepdims=True)
    centered = centers - centroid

    # SVD on 3xN centered points
    U, _, _ = np.linalg.svd(centered)

    normal = U[:, 2]
    normal /= np.linalg.norm(normal)

    if orient_towards is not None:
        orient_towards = np.asarray(orient_towards, dtype=float)

        if orient_towards.ndim == 1:
            orient_towards = orient_towards[:, np.newaxis]

        ref_vec = orient_towards.mean(axis=1) - centroid.squeeze()

        if np.dot(normal, ref_vec) < 0:
            normal = -normal

        U[:, 2] = normal

    return centroid, U


def compute_rigid_transform(A, B):
    """
    Computes the rotation R and translation t such that:
    B ≈ R @ A + t
    where A and B are 3×N matrices (N=4 for 4 tag corners)
    """
    assert A.shape == B.shape

    # Subtract centroids
    centroid_A = A.mean(axis=1, keepdims=True)
    centroid_B = B.mean(axis=1, keepdims=True)

    AA = A - centroid_A
    BB = B - centroid_B

    # Compute covariance matrix
    H = AA @ BB.T

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Correct possible reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    return R, t
