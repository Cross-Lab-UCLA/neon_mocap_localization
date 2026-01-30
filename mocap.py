import numpy as np
import open3d as o3d

from pose import Pose
from rigid import get_plane_coordinate_system


class MocapIRMarker:
    """
    Holds the timeseries of positions of an IR Marker
    """

    def __init__(self, Xs, Ys, Zs, id):
        self.Xs = Xs
        self.Ys = Ys
        self.Zs = Zs

        self.position = np.array([Xs, Ys, Zs])

        self.id = id


class MocapHead:
    def __init__(self):
        self.markers = []
        self.poses = []

    def add_marker(self, marker):
        self.markers.append(marker)
        self.poses.append(
            Pose(
                position=np.array([marker.Xs, marker.Ys, marker.Zs]),
                rotation=np.eye(3),
            )
        )


class MocapAprilTag:
    def __init__(self, tag_id):
        self.markers = []
        self.center = np.array([0, 0, 0])
        self.tag_id = tag_id

    def add_marker(self, marker):
        self.markers.append(marker)

    def estimate_tag_center(self):
        pos = np.array(
            [
                [marker.Xs for marker in self.markers],
                [marker.Ys for marker in self.markers],
                [marker.Zs for marker in self.markers],
            ]
        )
        self.center = np.mean(pos, axis=1)

    def estimate_size(self):
        """
        Estimate tag size [m] from mocap data.
        """

        tag_marker1_pos = np.array(
            [
                self.markers[0].Xs,
                self.markers[0].Ys,
                self.markers[0].Zs,
            ]
        )
        tag_marker2_pos = np.array(
            [
                self.markers[1].Xs,
                self.markers[1].Ys,
                self.markers[1].Zs,
            ]
        )

        self.tag_size = np.sqrt(
            np.sum(
                np.power(
                    tag_marker1_pos - tag_marker2_pos,
                    2,
                )
            )
        )


class MocapSurface:
    def __init__(self):
        self.apriltags = []
        self.markers = []

    def add_apriltag(self, apriltag):
        self.apriltags.append(apriltag)

    def add_marker(self, marker):
        self.markers.append(marker)

    def construct_pose(self, ir_marker_radius, orient_towards=None):
        """
        Construct the estimated pose of the surface in mocap system.
        """

        xs, ys, zs = [], [], []
        if not len(self.apriltags) == 0:
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

            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.1, ransac_n=3, num_iterations=1000
            )
            [a, b, c, d] = plane_model

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
