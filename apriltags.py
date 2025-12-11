import numpy as np

import cv2
from pupil_apriltags import Detector

from pose import Pose


class AprilTags:
    def __init__(self, neon, tag_size, img):
        self.at_detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        self.K = neon.camera_matrix
        self.D = neon.dist_coeffs
        self.tag_size = tag_size

        self.good_detection = True
        self.reprojection_errors = []

        self.undist_frame = None
        self.tag_poses = []
        self.tag_corners = []
        self.tag_ids = []

        self.detect_tags(img)
        if self.good_detection:
            self.extract_tag_poses()

    def detect_tags(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        self.new_K, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, img_gray.shape[::-1], 1, img_gray.shape[::-1]
        )
        self.undist_frame = cv2.undistort(
            img_gray, self.K, self.D, None, newCameraMatrix=self.new_K
        )

        fx = self.new_K[0, 0]
        fy = self.new_K[1, 1]
        cx = self.new_K[0, 2]
        cy = self.new_K[1, 2]

        try:
            self.at_detection = self.at_detector.detect(
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

        # colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

        # image = self.undist_frame
        # for detection in self.at_detection:
        #     tag_id = detection.tag_id
        #     corners = detection.corners
        #     center = detection.center

        #     pts = corners.astype(np.int32).reshape((-1, 1, 2))
        #     cv2.polylines(image, [pts], True, (0, 255, 0), 2)

        #     for i in range(4):
        #         pt = tuple(corners[i].astype(int))

        #         cv2.circle(image, pt, 8, colors[i], -1)

        #         cv2.putText(
        #             image,
        #             str(i),
        #             (pt[0] + 10, pt[1] + 10),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.8,
        #             colors[i],
        #             2,
        #         )

        #         cv2.putText(
        #             image,
        #             f"ID: {tag_id}",
        #             (int(center[0]), int(center[1])),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.8,
        #             (0, 0, 255),
        #             2,
        #         )

        # cv2.imshow("Pupil Apriltags Visualization", image)
        # cv2.destroyAllWindows()

        # tag_points_3d = np.array(
        #     [
        #         [-self.tag_size / 2, self.tag_size / 2, 0],  # BL
        #         [self.tag_size / 2, self.tag_size / 2, 0],  # BR
        #         [self.tag_size / 2, -self.tag_size / 2, 0],  # TR
        #         [-self.tag_size / 2, -self.tag_size / 2, 0],  # TL
        #     ]
        # )
        tag_points_3d = np.array(
            [
                [self.tag_size / 2, -self.tag_size / 2, 0],  # TR
                [-self.tag_size / 2, -self.tag_size / 2, 0],  # TL
                [-self.tag_size / 2, self.tag_size / 2, 0],  # BL
                [self.tag_size / 2, self.tag_size / 2, 0],  # BR
            ]
        )

        for detection in self.at_detection:
            # SOLVEPNP_IPPE_SQUARE returns 2 solutions for rotation/position/error.
            # First one always has smallest error
            ok, (tag_rotation, _), (tag_position, _), (error, _) = cv2.solvePnPGeneric(
                tag_points_3d,
                detection.corners,
                self.K,
                self.D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )

            if not ok:
                self.good_detection = False
                return

            # if detection.tag_id == 0:
            self.reprojection_errors.append(error)

            # corners = detection.corners
            # corners = [np.roll(corner_array, 2, axis=1) for corner_array in corners]
            # self.tag_corners.append(corners)

            # 1 is BL
            # 2 is BR
            # 3 is TR
            # 4 is TL
            self.tag_corners.append(detection.corners)
            self.tag_ids.append(detection.tag_id)

    def extract_tag_poses(self):
        for detection in self.at_detection:
            tag_pose = Pose(
                position=detection.pose_t.flatten(),
                rotation=detection.pose_R,
            )

            self.tag_poses.append(tag_pose)
