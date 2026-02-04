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
        self.reference_pose_in_mocap = None
        self.transformed_pose_in_mocap = Pose(
            position=np.zeros(1), rotation=np.zeros(1)
        )

    def set_pose(self, pose):
        self.pose = pose

    def calculate_reference_pose_in_mocap(
        self,
        surface_pose_mocap,
    ):
        if self.pose is None:
            raise ValueError("Pose in surface coordinates is not set.")

        self.reference_pose_in_mocap = surface_pose_mocap.apply(self.pose)

        self.reference_camera_axis_in_mocap = (
            self.reference_pose_in_mocap.rotation @ np.array([[0], [0], [1.0]])
        )
        self.reference_camera_axis_in_mocap = (
            self.reference_camera_axis_in_mocap.flatten()
        )

    def update_neon_camera_pose(
        self, transformed_head_marker_constellation, mocap_head
    ):
        transformation = mocap_head.get_relative_pose(
            transformed_head_marker_constellation
        )
        if transformation is None:
            return False

        # neon_pcd = o3d.geometry.PointCloud()
        # neon_pcd.points = o3d.utility.Vector3dVector(
        # self.reference_pose_in_mocap.position.reshape(-1, 3)
        # )

        # self.transformed_pose_in_mocap.position = np.asarray(
        #     neon_pcd.transform(transformation).points
        # ).flatten()

        R = transformation[:3, :3]
        t = transformation[:3, 3]

        self.transformed_pose_in_mocap.position = (
            R @ self.reference_pose_in_mocap.position
        ) + t

        self.transformed_pose_in_mocap.rotation = R

        self.transformed_camera_axis_in_mocap = R @ self.reference_camera_axis_in_mocap

        return True
