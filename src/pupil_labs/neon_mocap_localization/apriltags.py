import cv2
import numpy as np
import numpy.typing as npt
from pupil_apriltags import Detector  # type: ignore

from pupil_labs.neon_mocap_localization.neon import Neon
from pupil_labs.neon_mocap_localization.pose import Pose


class AprilTags:
    def __init__(
        self,
        detector: Detector,
        neon: Neon,
        tag_size: float,
        img: npt.NDArray[np.uint8],
        tag_corner_coordinates: dict[str, npt.NDArray[np.float64]],
        apriltags_to_use: list[str],
        surface_gaze_object_pts: npt.NDArray[np.floating] | None = None,
        surface_gaze_image_pts: npt.NDArray[np.floating] | None = None,
    ):
        self.detector = detector
        self.K = neon.camera_matrix
        self.D = neon.dist_coeffs
        self.tag_size = tag_size
        self.tag_corner_coordinates = tag_corner_coordinates
        self.apriltags_to_use = apriltags_to_use

        self.surface_gaze_object_pts = surface_gaze_object_pts
        self.surface_gaze_image_pts = surface_gaze_image_pts

        self.good_detection = False
        self.error = np.inf

        # self.pose = None
        # self.undist_frame = None

        self.detect_tags_and_extract_pose(img)

    def detect_tags_and_extract_pose(self, image: npt.NDArray[np.uint8]) -> None:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        self.new_K, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, img_gray.shape[::-1], 1, img_gray.shape[::-1]
        )
        self.undist_frame = cv2.undistort(
            img_gray, self.K, self.D, None, newCameraMatrix=self.new_K
        )

        # self.undist_frame = cv2.undistort(img_gray, self.K, self.D)

        fx: float = self.new_K[0, 0]
        fy: float = self.new_K[1, 1]
        cx: float = self.new_K[0, 2]
        cy: float = self.new_K[1, 2]

        # fx = self.K[0, 0]
        # fy = self.K[1, 1]
        # cx = self.K[0, 2]
        # cy = self.K[1, 2]

        try:
            self.at_detection = self.detector.detect(
                self.undist_frame,
                estimate_tag_pose=True,
                tag_size=self.tag_size,
                camera_params=(fx, fy, cx, cy),
            )
        except Exception:
            self.good_detection = False
            return

        if len(self.at_detection) < len(self.apriltags_to_use):  # type: ignore
            self.good_detection = False
            return

        zeroed_D = np.zeros((5, 1), dtype=np.float32)

        at_tag_pts = np.zeros((len(self.apriltags_to_use), 4, 2), dtype=np.float32)
        c = 0
        for detection in self.at_detection:  # type: ignore
            if str(detection.tag_id) in self.apriltags_to_use:
                # at_tag_pts[str(detection.tag_id)] = detection.corners
                at_tag_pts[c] = detection.corners
                c += 1

        object_pts = []
        # for k, v in self.tag_corner_coordinates.items():
        for k in self.apriltags_to_use:
            v = self.tag_corner_coordinates[k]
            for r in v:
                object_pts.append(r)

        object_pts_np = np.array(object_pts)

        image_pts = at_tag_pts.reshape(-1, 2)
        object_pts_np = object_pts_np.reshape(-1, 3)

        if (
            self.surface_gaze_image_pts is not None
            and self.surface_gaze_object_pts is not None
        ):
            ok, tag_rotation, tag_position, error = cv2.solvePnPGeneric(
                objectPoints=self.surface_gaze_object_pts,
                imagePoints=self.surface_gaze_image_pts,
                cameraMatrix=self.new_K,
                distCoeffs=self.D,
                flags=cv2.SOLVEPNP_IPPE,
            )
        else:
            ok, tag_rotation, tag_position, error = cv2.solvePnPGeneric(
                objectPoints=object_pts_np,
                imagePoints=image_pts,
                cameraMatrix=self.new_K,
                distCoeffs=zeroed_D,
                flags=cv2.SOLVEPNP_IPPE,
            )

        if not ok:
            self.good_detection = False
            return
        else:
            self.good_detection = True

        tag_rotation, tag_position = cv2.solvePnPRefineVVS(
            object_pts_np,
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
