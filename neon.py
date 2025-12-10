import cv2
import numpy as np


class Neon:
    def __init__(self, recording):
        # if recording is not None:
        camera_matrix = recording.calibration.scene_camera_matrix
        distortion_coefficients = recording.calibration.scene_distortion_coefficients

        self.camera_matrix = camera_matrix
        self.dist_coeffs = distortion_coefficients.flatten()

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
