import numpy as np
import json


class Neon:
    def __init__(self, calib_path=None, recording=None):
        if recording is not None:
            camera_matrix = recording.calibration.scene_camera_matrix
            distortion_coefficients = (
                recording.calibration.scene_distortion_coefficients
            )

            self.camera_matrix = camera_matrix
            self.dist_coeffs = distortion_coefficients.flatten()
        elif calib_path is not None:
            scene_calib = []
            with open(calib_path, "r") as f:
                scene_calib = json.load(f)

            self.camera_matrix = np.array(scene_calib["camera_matrix"])
            self.dist_coeffs = np.array(
                scene_calib["distortion_coefficients"]
            ).flatten()
        else:
            raise Exception("Need a native recording path or calibratioon file path")

        self.pose_in_tags = []
        self.pose_in_surface = None
        self.pose_in_mocap = None

    def add_pose_in_tag(self, pose):
        self.pose_in_tags.append(pose)

    def set_pose_in_surface(self, pose):
        self.pose_in_surface = pose

    def calculate_pose_in_mocap(self, surface_pose_mocap, R_apriltag_to_mocap):
        if self.pose_in_surface is None:
            raise ValueError("Pose in surface coordinates is not set.")

        # convert neon pose in surface coordinates to mocap format
        neon_pose_in_surface_mocap = self.pose_in_surface.to_pupil_labs_mocap_format(
            R_apriltag_to_mocap
        )
        self.pose_in_mocap = surface_pose_mocap.apply(neon_pose_in_surface_mocap)
