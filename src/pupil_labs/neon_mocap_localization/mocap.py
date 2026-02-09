import numpy as np
import numpy.typing as npt
import open3d as o3d  # type: ignore

from pupil_labs.neon_mocap_localization.pose import Pose
from pupil_labs.neon_mocap_localization.rigid import get_plane_coordinate_system


class MocapIRMarker:
    """Holds the timeseries of positions of an IR Marker"""

    def __init__(self, Xs: float, Ys: float, Zs: float, marker_id: int | str):
        self.Xs = Xs
        self.Ys = Ys
        self.Zs = Zs

        self.position = np.array([Xs, Ys, Zs])

        self.marker_id = marker_id


class MocapHead:
    def __init__(self) -> None:
        self.markers: list[MocapIRMarker] = []
        # self.poses = []

        # self.origin = None
        # self.x_axis = None
        # self.y_axis = None
        # self.z_axis = None

    def add_marker(self, marker: MocapIRMarker) -> None:
        self.markers.append(marker)
        # self.poses.append(
        # Pose(
        # position=np.array([marker.Xs, marker.Ys, marker.Zs]),
        # rotation=np.eye(3),
        # )
        # )

    def get_local_coord_sys(self) -> None:
        # determine position of neon camera relative to frame markers
        neon_marker_positions_in_mocap = np.array([
            [ir_marker.Xs, ir_marker.Ys, ir_marker.Zs] for ir_marker in self.markers
        ]).T

        self.origin = np.nanmean(neon_marker_positions_in_mocap, axis=1)

        self.x_axis = neon_marker_positions_in_mocap[:, 0] - self.origin
        self.y_axis = neon_marker_positions_in_mocap[:, 1] - self.origin
        self.z_axis = neon_marker_positions_in_mocap[:, 2] - self.origin

        self.x_axis /= np.linalg.norm(self.x_axis)
        self.y_axis /= np.linalg.norm(self.y_axis)
        self.z_axis /= np.linalg.norm(self.z_axis)

        R = np.zeros((3, 3))

        R[:, 0] = self.x_axis
        R[:, 1] = self.y_axis
        R[:, 2] = self.z_axis

        self.pose = Pose(
            position=self.origin,
            rotation=R,
        )

    def get_relative_pose(
        self, marker_constellation: list[MocapIRMarker]
    ) -> npt.NDArray[np.float64] | None:
        reference_neon_marker_positions = np.array([
            [ir_marker.Xs, ir_marker.Ys, ir_marker.Zs]
            for ir_marker in self.markers
            if not np.isnan(ir_marker.Xs)
        ]).T

        new_neon_marker_positions = np.array([
            [ir_marker.Xs, ir_marker.Ys, ir_marker.Zs]
            for ir_marker in marker_constellation
            if not np.isnan(ir_marker.Xs)
        ]).T

        if np.isnan(new_neon_marker_positions).all():
            return None

        # 1. Prepare your points (N, 3) as Open3D PointClouds
        source_pcd = o3d.geometry.PointCloud()
        # breakpoint()
        source_pcd.points = o3d.utility.Vector3dVector(
            reference_neon_marker_positions.T
        )

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(new_neon_marker_positions.T)

        # 2. Define correspondences (index-to-index)
        # If points_A[0] matches points_B[0], the vector is [[0,0], [1,1], ...]
        corres = o3d.utility.Vector2iVector(
            np.arange(len(reference_neon_marker_positions.T))
            .reshape(-1, 1)
            .repeat(2, axis=1)
        )

        # 3. Estimate transformation (This is the Kabsch/Procrustes part)
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        transformation: npt.NDArray[np.float64] = estimator.compute_transformation(
            source_pcd, target_pcd, corres
        )

        return transformation


class MocapAprilTag:
    def __init__(self, tag_id: str):
        self.markers: list[MocapIRMarker] = []
        self.center = np.array([0, 0, 0])
        self.tag_id = tag_id

    def add_marker(self, marker: MocapIRMarker) -> None:
        self.markers.append(marker)

    def estimate_tag_center(self) -> None:
        pos = np.array([
            [marker.Xs for marker in self.markers],
            [marker.Ys for marker in self.markers],
            [marker.Zs for marker in self.markers],
        ])
        self.center = np.mean(pos, axis=1)

    def estimate_size(self) -> None:
        """Estimate tag size [m] from mocap data."""
        tag_marker1_pos = np.array([
            self.markers[0].Xs,
            self.markers[0].Ys,
            self.markers[0].Zs,
        ])
        tag_marker2_pos = np.array([
            self.markers[1].Xs,
            self.markers[1].Ys,
            self.markers[1].Zs,
        ])

        self.tag_size = np.sqrt(
            np.sum(
                np.power(
                    tag_marker1_pos - tag_marker2_pos,
                    2,
                )
            )
        )


class MocapSurface:
    def __init__(self) -> None:
        self.apriltags: list[MocapAprilTag] = []
        self.markers: list[MocapIRMarker] = []

    def add_apriltag(self, apriltag: MocapAprilTag) -> None:
        self.apriltags.append(apriltag)

    def add_marker(self, marker: MocapIRMarker) -> None:
        self.markers.append(marker)

    def construct_pose(
        self,
        ir_marker_radius: float,
        orient_towards: npt.NDArray[np.float64] | None = None,
    ) -> bool:
        """Construct the estimated pose of the surface in mocap system."""
        xs, ys, zs = [], [], []
        if len(self.apriltags) != 0:
            for apriltag in self.apriltags:
                for marker in apriltag.markers:
                    xs.append(marker.Xs)
                    ys.append(marker.Ys)
                    zs.append(marker.Zs)
        else:
            for marker in self.markers:
                xs.append(marker.Xs)
                ys.append(marker.Ys)
                zs.append(marker.Zs)

        poses = np.vstack([xs, ys, zs]).T

        self.centroid = np.mean(poses, axis=0)

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(poses)

            _, inliers = pcd.segment_plane(
                distance_threshold=0.004, ransac_n=3, num_iterations=1000
            )
            # [a, b, c, d] = plane_model

            inlier_cloud = np.asarray(pcd.select_by_index(inliers).points)

            (self.x_axis, self.y_axis, self.normal) = get_plane_coordinate_system(
                inlier_cloud
            )

            if orient_towards is not None:
                orient_towards = np.asarray(orient_towards, dtype=float)

                ref_vec = orient_towards - self.centroid.squeeze()

                if np.dot(self.normal, ref_vec) < 0:
                    self.normal = -self.normal

            R = np.zeros((3, 3))
            R[:, 0] = self.x_axis
            R[:, 1] = self.y_axis
            R[:, 2] = self.normal

            R[:, 0] /= np.linalg.norm(R[:, 0])
            R[:, 1] /= np.linalg.norm(R[:, 1])
            R[:, 2] /= np.linalg.norm(R[:, 2])
        except Exception:
            return False

        self.pose = Pose(
            position=self.centroid.flatten() - R[:, 2] * ir_marker_radius,
            rotation=R,
        )

        return True

    def construct_pose_simple(
        self,
        ir_marker_radius: float,
        orient_towards: npt.NDArray[np.float64] | None = None,
    ) -> bool:
        """Construct the estimated pose of the surface in mocap system."""
        xs, ys, zs = [], [], []
        if len(self.apriltags) != 0:
            for apriltag in self.apriltags:
                for marker in apriltag.markers:
                    xs.append(marker.Xs)
                    ys.append(marker.Ys)
                    zs.append(marker.Zs)
        else:
            for marker in self.markers:
                xs.append(marker.Xs)
                ys.append(marker.Ys)
                zs.append(marker.Zs)

        poses = np.vstack([xs, ys, zs]).T

        self.centroid = np.mean(poses, axis=0)

        try:
            self.x_axis = np.array([
                self.markers[0].Xs,
                self.markers[0].Ys,
                self.markers[0].Zs,
            ]) - np.array([self.markers[1].Xs, self.markers[1].Ys, self.markers[1].Zs])

            self.y_axis = np.array([
                self.markers[0].Xs,
                self.markers[0].Ys,
                self.markers[0].Zs,
            ]) - np.array([self.markers[3].Xs, self.markers[3].Ys, self.markers[3].Zs])

            self.x_axis /= np.linalg.norm(self.x_axis)
            self.y_axis /= np.linalg.norm(self.y_axis)

            self.normal = np.cross(self.x_axis, self.y_axis)

            if orient_towards is not None:
                orient_towards = np.asarray(orient_towards, dtype=float)

                ref_vec = orient_towards - self.centroid.squeeze()

                if np.dot(self.normal, ref_vec) < 0:
                    self.normal = -self.normal

            self.normal /= np.linalg.norm(self.normal)

            R = np.zeros((3, 3))
            R[:, 0] = self.x_axis
            R[:, 1] = self.y_axis
            R[:, 2] = self.normal
        except Exception:
            return False

        self.pose = Pose(
            position=self.centroid.flatten() - R[:, 2] * ir_marker_radius,
            rotation=R,
        )

        return True
