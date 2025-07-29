import numpy as np

from pose import Pose
from rigid import compute_rigid_transform, fit_plane


class Surface:
    def __init__(self, tag_size):
        self.tag_size = tag_size
        self.tag_poses = []
        self.surface_corners = []

    def add_tag_pose(self, pose):
        self.tag_poses.append(pose)
        self.surface_corners.append(pose.position)

    def build_surface(self):
        centers = np.stack([pose.position for pose in self.tag_poses], axis=1)

        centroid, R = fit_plane(centers)

        self.x_axis = R[:, 0]  # first column = x-axis of best-fit plane
        self.y_axis = R[:, 1]  # second column = y-axis of best-fit
        self.normal = R[:, 2]  # third column = normal to best-fit plane

        self.pose_in_neon = Pose(
            position=centroid.flatten(),
            rotation=R,
        )

    def estimate_pose_from_corners(self):
        # Example: local and global corner arrays, shape (3, 4)
        # Replace with your real data
        s = self.tag_size  # Tag size in meters
        local_corners = np.array(
            [
                [-s / 2, s / 2, s / 2, -s / 2],
                [s / 2, s / 2, -s / 2, -s / 2],
                [0, 0, 0, 0],
            ]
        )

        x0, y0, z0 = self.tag_poses[2].position  # top left
        x1, y1, z1 = self.tag_poses[1].position  # top right
        x2, y2, z2 = self.tag_poses[0].position  # bottom right
        x3, y3, z3 = self.tag_poses[3].position  # bottom left

        global_corners = np.array(
            [[x0, x1, x2, x3], [y0, y1, y2, y3], [z0, z1, z2, z3]]
        )

        R, t = compute_rigid_transform(local_corners, global_corners)

        return Pose(
            position=t.flatten(),
            rotation=R,
        )
