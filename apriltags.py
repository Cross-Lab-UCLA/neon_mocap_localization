import cv2
from pupil_apriltags import Detector

from pose import Pose


class AprilTags:
    def __init__(self, camera_matrix, dist_coeffs, tag_size):
        self.at_detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )
        self.K = camera_matrix
        self.D = dist_coeffs
        self.tag_size = tag_size

        self.tag_poses = []
        self.tag_corners = []

    def detect_tags(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        self.new_K, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, img_gray.shape[::-1], 1, img_gray.shape[::-1]
        )
        undist_frame = cv2.undistort(
            img_gray, self.K, self.D, None, newCameraMatrix=self.new_K
        )

        fx = self.new_K[0, 0]
        fy = self.new_K[1, 1]
        cx = self.new_K[0, 2]
        cy = self.new_K[1, 2]

        self.at_detection = self.at_detector.detect(
            undist_frame,
            estimate_tag_pose=True,
            tag_size=self.tag_size,
            camera_params=[fx, fy, cx, cy],
        )

        for detection in self.at_detection:
            self.tag_corners.append(detection.corners)

    def extract_tag_poses(self):
        for detection in self.at_detection:
            tag_pose = Pose(
                position=detection.pose_t.flatten(),
                rotation=detection.pose_R,
            )

            self.tag_poses.append(tag_pose)
