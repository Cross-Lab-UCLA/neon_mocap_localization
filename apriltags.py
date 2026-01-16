import cv2
import numpy as np

from pose import Pose


class AprilTags:
    def __init__(self, detector, neon, tag_size, img, tag_corner_coordinates):
        self.detector = detector
        self.K = neon.camera_matrix
        self.D = neon.dist_coeffs
        self.tag_size = tag_size
        self.tag_corner_coordinates = tag_corner_coordinates

        self.good_detection = True
        self.error = np.inf

        self.pose = None

        self.undist_frame = None

        self.detect_tags_and_extract_pose(img)

    def detect_tags_and_extract_pose(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # self.new_K, _ = cv2.getOptimalNewCameraMatrix(
        #     self.K, self.D, img_gray.shape[::-1], 1, img_gray.shape[::-1]
        # )
        # self.undist_frame = cv2.undistort(
        #     img_gray, self.K, self.D, None, newCameraMatrix=self.new_K
        # )

        self.undist_frame = cv2.undistort(img_gray, self.K, self.D)

        # fx = self.new_K[0, 0]
        # fy = self.new_K[1, 1]
        # cx = self.new_K[0, 2]
        # cy = self.new_K[1, 2]

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

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

        if len(self.at_detection) < 4:
            self.good_detection = False
            return

        zeroed_D = np.zeros((5, 1), dtype=np.float32)

        at_tag_pts = np.zeros((4, 4, 2), dtype=np.float32)
        for detection in self.at_detection:
            at_tag_pts[detection.tag_id] = detection.corners

        object_pts = []
        for v in self.tag_corner_coordinates.values():
            for r in v:
                object_pts.append(r)

        object_pts = np.array(object_pts)

        image_pts = at_tag_pts.reshape(-1, 2)
        object_pts = object_pts.reshape(-1, 3)

        ok, tag_rotation, tag_position, error = cv2.solvePnPGeneric(
            objectPoints=object_pts,
            imagePoints=image_pts,
            cameraMatrix=self.K,
            distCoeffs=zeroed_D,
            flags=cv2.SOLVEPNP_SQPNP,
        )

        if not ok:
            self.good_detection = False
            return

        # ok, tag_rotation, tag_position, _ = cv2.solvePnPRansac(
        #     objectPoints=object_pts,
        #     imagePoints=image_pts,
        #     cameraMatrix=self.K,
        #     distCoeffs=zeroed_D,
        #     rvec=tag_rotation[0],
        #     tvec=tag_position[0],
        #     useExtrinsicGuess=True,
        # )

        # if not ok:
        #     self.good_detection = False
        #     return

        tag_rotation, tag_position = cv2.solvePnPRefineVVS(
            object_pts,
            image_pts,
            self.K,
            zeroed_D,
            tag_rotation[0],
            tag_position[0],
        )

        rotation_matrix, _ = cv2.Rodrigues(tag_rotation)

        rotation_matrix[:, 2] *= -1

        self.error = error[0]

        self.pose = Pose(
            tag_position.flatten(),
            rotation_matrix,
        )
