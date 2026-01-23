import cv2
import matplotlib.pyplot as plt
import numpy as np

from pose import Pose


class AprilTags:
    def __init__(
        self,
        detector,
        neon,
        tag_size,
        img,
        tag_corner_coordinates,
        apriltags_to_use,
        surface_gaze_object_pts=None,
        surface_gaze_image_pts=None,
    ):
        self.detector = detector
        self.K = neon.camera_matrix
        self.D = neon.dist_coeffs
        self.tag_size = tag_size
        self.tag_corner_coordinates = tag_corner_coordinates
        self.apriltags_to_use = apriltags_to_use

        self.surface_gaze_object_pts = surface_gaze_object_pts
        self.surface_gaze_image_pts = surface_gaze_image_pts

        self.good_detection = True
        self.error = np.inf

        self.pose = None

        self.undist_frame = None

        self.detect_tags_and_extract_pose(img)

    def detect_tags_and_extract_pose(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        self.new_K, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, img_gray.shape[::-1], 1, img_gray.shape[::-1]
        )
        self.undist_frame = cv2.undistort(
            img_gray, self.K, self.D, None, newCameraMatrix=self.new_K
        )

        # self.undist_frame = cv2.undistort(img_gray, self.K, self.D)

        fx = self.new_K[0, 0]
        fy = self.new_K[1, 1]
        cx = self.new_K[0, 2]
        cy = self.new_K[1, 2]

        # fx = self.K[0, 0]
        # fy = self.K[1, 1]
        # cx = self.K[0, 2]
        # cy = self.K[1, 2]

        try:
            self.at_detection = self.detector.detect(
                self.undist_frame,
                estimate_tag_pose=True,
                tag_size=self.tag_size,
                camera_params=[fx, fy, cx, cy],
            )
        except Exception:
            self.good_detection = False
            return

        if len(self.at_detection) < len(self.apriltags_to_use):
            self.good_detection = False
            return

        zeroed_D = np.zeros((5, 1), dtype=np.float32)

        at_tag_pts = np.zeros((len(self.apriltags_to_use), 4, 2), dtype=np.float32)
        for detection in self.at_detection:
            if detection.tag_id in self.apriltags_to_use:
                at_tag_pts[detection.tag_id] = detection.corners

        object_pts = []
        for k, v in self.tag_corner_coordinates.items():
            for r in v:
                object_pts.append(r)

        object_pts = np.array(object_pts)

        image_pts = at_tag_pts.reshape(-1, 2)
        object_pts = object_pts.reshape(-1, 3)

        if (
            self.surface_gaze_image_pts is not None
            and self.surface_gaze_object_pts is not None
        ):
            object_pts = self.surface_gaze_object_pts
            image_pts = self.surface_gaze_image_pts

            ok, tag_rotation, tag_position, error = cv2.solvePnPGeneric(
                objectPoints=object_pts,
                imagePoints=image_pts,
                cameraMatrix=self.new_K,
                distCoeffs=self.D,
                flags=cv2.SOLVEPNP_IPPE,
            )
        else:
            ok, tag_rotation, tag_position, error = cv2.solvePnPGeneric(
                objectPoints=object_pts,
                imagePoints=image_pts,
                cameraMatrix=self.new_K,
                distCoeffs=zeroed_D,
                flags=cv2.SOLVEPNP_IPPE,
            )

        if not ok:
            self.good_detection = False
            return

        tag_rotation, tag_position = cv2.solvePnPRefineVVS(
            object_pts,
            image_pts,
            self.new_K,
            zeroed_D,
            tag_rotation[0],
            tag_position[0],
        )

        rotation_matrix, _ = cv2.Rodrigues(tag_rotation)

        self.error = error[0]

        self.pose = Pose(
            tag_position.flatten(),
            rotation_matrix,
        )
