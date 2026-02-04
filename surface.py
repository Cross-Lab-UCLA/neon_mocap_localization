import numpy as np


class Surface:
    def __init__(self, tag_size):
        self.tag_size = tag_size
        self.pose = None

        self.surface_corners = []
        self.local_coords = None

        self.pose_in_neon = None

        self.x_axis = None
        self.y_axis = None
        self.normal = None

    def set_pose(self, pose):
        self.pose = pose

        self.x_axis = pose.rotation[:, 0]
        self.y_axis = pose.rotation[:, 1]
        self.normal = pose.rotation[:, 2]

        self.x_axis /= np.linalg.norm(self.x_axis)
        self.y_axis /= np.linalg.norm(self.y_axis)
        self.normal /= np.linalg.norm(self.normal)
