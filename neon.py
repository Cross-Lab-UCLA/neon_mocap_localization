import cv2
import numpy as np

from pose import Pose


class Neon:
    def __init__(self, recording=None, K=None, D=None):
        if recording is None:
            camera_matrix = K
            distortion_coefficients = D
        else:
            camera_matrix = recording.calibration.scene_camera_matrix
            distortion_coefficients = (
                recording.calibration.scene_distortion_coefficients
            )

        self.camera_matrix = camera_matrix
        self.dist_coeffs = distortion_coefficients.flatten()

        self.pose = None
        self.pose_in_mocap = None

    def set_pose(self, pose):
        self.pose = pose

    def _convert_to_mocap_format(self, pose):
        R = np.array(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1],
            ]
        )

        return Pose(
            position=R @ pose.position,
            rotation=R @ pose.rotation,
        )

    def calculate_pose_in_mocap(
        self,
        surface_pose_mocap,
    ):
        if self.pose is None:
            raise ValueError("Pose in surface coordinates is not set.")

        self.pose_in_surface_mocap = self._convert_to_mocap_format(self.pose)
        self.pose_in_mocap = surface_pose_mocap.apply(self.pose_in_surface_mocap)
