import numpy as np

from pose import Pose
from rigid import fit_plane_simple


class Surface:
    def __init__(self, tag_size):
        self.tag_size = tag_size
        self.tag_poses = []
        self.surface_corners = []
        self.marker_local_coords = None

    def add_tag_pose(self, pose):
        self.tag_poses.append(pose)
        self.surface_corners.append(pose.position)

    def build_surface(self, orient_towards, from_poses=True):
        # centers = np.stack([pose.position for pose in self.tag_poses], axis=1)
        if not from_poses:
            tag_half = self.tag_size / 2
            tag_plane = np.array(
                [
                    [-tag_half, -tag_half, 0],
                    [tag_half, -tag_half, 0],
                    [tag_half, tag_half, 0],
                    [-tag_half, tag_half, 0],
                    [-tag_half, -tag_half, 0],
                ]
            )

            centers = np.zeros((len(self.tag_poses), 5, 3))
            for i, tag_pose in enumerate(self.tag_poses):
                # Transform tag corners to camera frame using tag_pose
                tag_corners_in_cam = (
                    tag_pose.rotation @ tag_plane.T
                ).T + tag_pose.position

                for c in range(5):
                    centers[i, c] = tag_corners_in_cam[c]

            centers.shape = (-1, 3)

        try:
            centroid = np.mean(np.array(self.surface_corners), axis=0)

            # top right - top left
            self.x_axis = self.surface_corners[1] - self.surface_corners[0]
            self.x_axis /= np.linalg.norm(self.x_axis)

            # top left - bottom left
            self.y_axis = -(self.surface_corners[3] - self.surface_corners[0])
            self.y_axis /= np.linalg.norm(self.y_axis)

            # cross-product of x & y
            self.normal = np.cross(self.x_axis, self.y_axis)
            self.normal /= np.linalg.norm(self.normal)

            if orient_towards is not None:
                orient_towards = np.asarray(orient_towards, dtype=float)

                ref_vec = orient_towards - centroid.squeeze()

                if np.dot(self.normal, ref_vec) < 0:
                    self.normal = -self.normal

            R = np.zeros((3, 3))
            R[:, 0] = self.x_axis
            R[:, 1] = self.y_axis
            R[:, 2] = self.normal
        except Exception:
            return False

        print("huh")

        self.pose_in_neon = Pose(
            position=centroid.flatten(),
            rotation=R,
        )

        return True

    def build_surface_simple(self, orient_towards):
        centers = np.stack([pose.position for pose in self.tag_poses], axis=1)

        try:
            centroid, R = fit_plane_simple(centers, orient_towards)
        except Exception:
            return False

        self.x_axis = R[:, 0]  # first column = x-axis of best-fit plane
        self.y_axis = R[:, 1]  # second column = y-axis of best-fit
        self.normal = R[:, 2]  # third column = normal to best-fit plane

        self.pose_in_neon = Pose(
            position=centroid.flatten(),
            rotation=R,
        )

        return True
