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

        self.pose_in_tags = []
        self.pose_in_surface = None
        self.pose_in_mocap = None

    def calculate_pose_in_mocap(
        self,
        mocap_surface,
        neon_apriltags,
    ):
        K = neon_apriltags.K
        D = neon_apriltags.D

        apriltag_corners = [
            [[corner[0], corner[1]] for corner in tag_corners]
            for tag_corners in neon_apriltags.tag_corners
        ]

        image_points_2d = np.array(apriltag_corners)
        # print(image_points_2d.shape)

        mocap_corners = [
            [[marker.Xs, marker.Ys, marker.Zs] for marker in apriltag.markers]
            for apriltag in mocap_surface.apriltags
        ]

        mocap_points_3d = np.array(mocap_corners)
        # print("yo")
        # print(mocap_points_3d.shape)

        image_pts = image_points_2d.reshape(-1, 2)
        object_pts = mocap_points_3d.reshape(-1, 3)
        # print(image_pts.shape)
        # print(object_pts.shape)

        image_pts_safe = np.ascontiguousarray(image_pts, dtype=np.float64)
        object_pts_safe = np.ascontiguousarray(object_pts, dtype=np.float64)
        K_safe = np.ascontiguousarray(K, dtype=np.float64)
        D_safe = np.ascontiguousarray(D, dtype=np.float64)

        try:
            ok, rvecs, tvecs = cv2.solvePnP(
                object_pts_safe,
                image_pts_safe,
                K_safe,
                D_safe,
                flags=cv2.SOLVEPNP_SQPNP,
            )

            # ok, rvecs, tvecs, rmses = cv2.solvePnPGeneric(
            #     object_pts_safe,
            #     image_pts_safe,
            #     K_safe,
            #     D_safe,
            #     flags=cv2.SOLVEPNP_ITERATIVE,
            # )
            if not ok:
                return None

            if len(rvecs) == 2:
                (best_rvec, _) = rvecs
                (best_tvec, _) = tvecs
                (rmse, _) = rmses
            elif len(rvecs) == 1:
                best_rvec = rvecs[0]
                best_tvec = tvecs[0]
                rmse = rmses[0]
            else:
                best_rvec = rvecs
                best_tvec = tvecs

            if best_rvec is None or best_tvec is None:
                return None
        except Exception as e:
            print(e)
            return None

        best_rvec, best_tvec = cv2.solvePnPRefineLM(
            object_pts_safe, image_pts_safe, K_safe, D_safe, best_rvec, best_tvec
        )

        # R_cam_to_world = R_world_to_cam.T
        R_mat, _ = cv2.Rodrigues(best_rvec)
        best_R_inv = R_mat.T

        # Pos = -R^T * t
        best_cam_pos_world = -best_R_inv @ best_tvec

        proj_pts, _ = cv2.projectPoints(
            object_pts_safe,
            best_rvec,
            best_tvec,
            K_safe,
            D_safe,
        )
        error = cv2.norm(image_pts_safe, proj_pts.squeeze(), cv2.NORM_L2)
        rmse = error / np.sqrt(len(image_pts))

        self.pose_in_mocap = Pose(
            position=best_cam_pos_world.flatten(),
            rotation=best_R_inv,
        )

        return rmse
