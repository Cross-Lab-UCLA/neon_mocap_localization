import numpy as np

import pupil_labs.neon_recording as plnr
from pupil_labs.neon_mocap_localization.cloud_recording import CloudRecording
from pupil_labs.neon_mocap_localization.mocap import MocapHead, MocapIRMarker
from pupil_labs.neon_mocap_localization.pose import Pose


class Neon:
    def __init__(
        self,
        recording: plnr.NeonRecording | CloudRecording,
    ):
        if recording.calibration is not None:
            camera_matrix = recording.calibration.scene_camera_matrix
            distortion_coefficients = (
                recording.calibration.scene_distortion_coefficients
            )
        else:
            raise ValueError("Recording contains no camera calibration data")

        self.camera_matrix = camera_matrix
        self.dist_coeffs = distortion_coefficients.flatten()

        # self.pose = None
        # self.reference_pose_in_mocap = None
        self.transformed_pose_in_mocap = Pose(
            position=np.zeros(1), rotation=np.zeros(1)
        )

    def set_pose(self, pose: Pose) -> None:
        self.pose = pose

    def calculate_reference_pose_in_mocap(
        self,
        surface_pose_mocap: Pose,
    ) -> None:
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
        self,
        transformed_head_marker_constellation: list[MocapIRMarker],
        mocap_head: MocapHead,
    ) -> bool:
        transformation = mocap_head.get_relative_pose(
            transformed_head_marker_constellation
        )
        if transformation is None:
            return False

        R = transformation[:3, :3]
        t = transformation[:3, 3]

        self.transformed_pose_in_mocap.position = (
            R @ self.reference_pose_in_mocap.position
        ) + t

        self.transformed_pose_in_mocap.rotation = R

        self.transformed_camera_axis_in_mocap = R @ self.reference_camera_axis_in_mocap

        return True
