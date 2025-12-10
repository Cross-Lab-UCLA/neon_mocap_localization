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

    # def calculate_pose_in_mocap(
    #     self, mocap_surface, mocap_points_3d, apriltag_corners, K, D
    # ):
    #     image_points_2d = np.array(apriltag_corners)

    #     # mocap_points_3d = mocap_points_3d[::-1]
    #     # image_points_2d = image_points_2d[::-1]

    #     image_points_2d = image_points_2d.reshape(-1, 2)
    #     mocap_points_3d = mocap_points_3d.reshape(-1, 3)

    #     object_pts = np.ascontiguousarray(mocap_points_3d, dtype=np.float64)
    #     image_pts = np.ascontiguousarray(image_points_2d, dtype=np.float64)

    #     try:
    #         # ok, rvec, tvec = cv2.solvePnP(
    #         #     object_pts,
    #         #     image_pts,
    #         #     neon_apriltags.new_K,
    #         #     neon_apriltags.D,
    #         #     flags=cv2.SOLVEPNP_SQPNP,
    #         # )

    #         ok, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
    #             object_pts,
    #             image_pts,
    #             K,
    #             D,
    #             flags=cv2.SOLVEPNP_IPPE_SQUARE,
    #         )
    #         if not ok:
    #             return None, None, None
    #     except Exception:
    #         return None, None, None

    #     # best_rvec, best_tvec = cv2.solvePnPRefineLM(
    #     #     object_pts, image_pts, neon_apriltags.new_K, neon_apriltags.D, rvec, tvec
    #     # )

    #     # R_cam_to_world = R_world_to_cam.T
    #     # R_mat, _ = cv2.Rodrigues(best_rvec)
    #     # best_R_inv = R_mat.T

    #     # Pos = -R^T * t
    #     # best_cam_pos_world = -best_R_inv @ best_tvec

    #     # best_cam_pos_world = np.array(
    #     # [best_cam_pos_world[0], best_cam_pos_world[2], best_cam_pos_world[1]]
    #     # )

    #     best_rvec = None
    #     best_tvec = None
    #     best_cam_pos_world = None
    #     best_R_inv = None

    #     best_idx = np.argmin(reproj_errs)
    #     # Iterate through solutions (usually 2)
    #     # for i in range(len(rvecs)):
    #     r = rvecs[best_idx]
    #     t = tvecs[best_idx]

    #     rvec_final, tvec_final = cv2.solvePnPRefineLM(object_pts, image_pts, K, D, r, t)

    #     # R_cam_to_world = R_world_to_cam.T
    #     R_mat, _ = cv2.Rodrigues(rvec_final)
    #     R_inv = R_mat.T

    #     # Pos = -R^T * t
    #     cam_pos_world = -R_inv @ tvec_final

    #     # print(cam_pos_world.shape, mocap_surface.pose.position.shape)

    #     # if not enforce_camera_facing_side(
    #     # cam_pos_world, mocap_surface.pose.position, mocap_surface.normal
    #     # ):
    #     best_rvec = r
    #     best_tvec = t
    #     best_cam_pos_world = cam_pos_world
    #     best_R_inv = R_inv
    #     # else:
    #     #     # Vector from Plane Point to Camera
    #     #     v = cam_pos_world.squeeze() - mocap_surface.pose.position.squeeze()

    #     #     # Projected distance onto the normal
    #     #     dist = np.dot(v.squeeze(), mocap_surface.normal.squeeze())

    #     #     # The Reflection Formula: P_new = P_old - 2 * dist * normal
    #     #     new_pos = (
    #     #         cam_pos_world.squeeze() - 2 * dist * mocap_surface.normal.squeeze()
    #     #     )

    #     #     best_rvec = r
    #     #     best_tvec = t
    #     #     best_cam_pos_world = new_pos
    #     #     best_R_inv = R_inv

    #     if best_rvec is None:
    #         return None, None, None

    #     proj_pts, _ = cv2.projectPoints(
    #         object_pts,
    #         best_rvec,
    #         best_tvec,
    #         K,
    #         D,
    #     )
    #     error = cv2.norm(image_pts, proj_pts.squeeze(), cv2.NORM_L2)
    #     rmse = error / np.sqrt(len(image_pts))

    #     return best_cam_pos_world.flatten(), best_R_inv, rmse

    # def calculate_pose_in_mocap(self, surface_pose_mocap, R_apriltag_to_mocap):
    def calculate_pose_in_mocap(self, surface_pose_mocap):
        # convert neon pose in surface coordinates to mocap format
        # neon_pose_in_surface_mocap = self.pose_in_surface.to_pupil_labs_mocap_format(
        # R_apriltag_to_mocap
        # )
        # self.pose_in_mocap = surface_pose_mocap.apply(neon_pose_in_surface_mocap)

        self.pose_in_mocap = surface_pose_mocap.apply(self.pose_in_surface)


def enforce_camera_facing_side(camera_pos_world, tag_center_world, tag_normal_world):
    """
    Checks if the camera is on the correct side of the planar surface.
    If not, it suggests the pose was flipped.
    """
    # 1. Vector from Tag Surface to Camera
    # vec_tag_to_cam = camera_pos_world - tag_center_world
    vec_tag_to_cam = camera_pos_world.squeeze() - tag_center_world.squeeze()

    # 2. Check alignment with the Surface Normal
    # Dot Product > 0 means they point in generally the same direction (Correct)
    # Dot Product < 0 means they point opposite (Camera is 'behind' the wall)
    dot_prod = np.dot(vec_tag_to_cam, tag_normal_world)

    # print(camera_pos_world.shape, tag_center_world.shape)

    if dot_prod < 0:
        # print(f"⚠️ FLIP DETECTED: Camera is on the wrong side (Dot = {dot_prod:.2f})")
        return True  # It is flipped
    else:
        return False  # It is correct
