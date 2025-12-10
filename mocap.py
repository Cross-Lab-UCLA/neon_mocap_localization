import numpy as np

from apriltags import AprilTags
from pose import Pose
from rigid import fit_plane
from surface import Surface


class MocapIRMarker:
    """
    Holds the timeseries of positions of an IR Marker
    """

    def __init__(self, Xs, Ys, Zs):
        self.Xs = Xs
        self.Ys = Ys
        self.Zs = Zs


class MocapHead:
    def __init__(self):
        self.markers = []

    def add_marker(self, marker):
        self.markers.append(marker)


class MocapAprilTag:
    def __init__(self):
        self.markers = []

    def add_marker(self, marker):
        self.markers.append(marker)

    def estimate_size_mm(self):
        """
        Estimate tag size [m] from mocap data.
        """

        # nsamples = len(self.markers[0].Xs)
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

    def add_apriltag(self, apriltag):
        self.apriltags.append(apriltag)

    def construct_pose(self, orient_towards=None):
        """
        Construct the estimated pose of the surface in mocap system.
        """

        apriltag = []
        xs, ys, zs = [], [], []
        for apriltag in self.apriltags:
            for marker in apriltag.markers:
                xs.append(marker.Xs)
                ys.append(marker.Ys)
                zs.append(marker.Zs)

        centers = np.array(
            [
                [apriltag.center[0] for apriltag in self.apriltags],
                [apriltag.center[1] for apriltag in self.apriltags],
                [apriltag.center[2] for apriltag in self.apriltags],
            ]
        )

        centroid = np.mean(centers, axis=1)

        self.x_axis = self.apriltags[1].center - self.apriltags[0].center
        self.x_axis /= np.linalg.norm(self.x_axis)

        self.y_axis = self.apriltags[3].center - self.apriltags[0].center
        self.y_axis /= np.linalg.norm(self.y_axis)

        self.normal = np.cross(self.x_axis, self.y_axis)
        self.normal /= np.linalg.norm(self.normal)

        if orient_towards is not None:
            orient_towards = np.asarray(orient_towards, dtype=float)

            ref_vec = orient_towards - centroid.squeeze()

            if np.dot(self.normal, ref_vec) < 0:
                self.normal = -self.normal

        rotation = np.zeros((3, 3))
        rotation[:, 0] = self.x_axis
        rotation[:, 1] = self.y_axis
        rotation[:, 2] = self.normal

        self.pose = Pose(
            position=centroid.flatten(),
            rotation=rotation,
        )

        self.apriltags[0].estimate_size_mm()
        self.tag_size = self.apriltags[0].tag_size

        return True


def extract_apriltag_surface(neon, mocap_surface, img):
    # detect apriltags in neon image

    neon_apriltags = AprilTags(neon, 130, img)
    # neon_apriltags = AprilTags(neon, mocap_surface.tag_size, img)

    if not neon_apriltags.good_detection:
        return None, None

    # find neon's pose in each apriltag coordinate system
    for pose in neon_apriltags.tag_poses:
        neon.add_pose_in_tag(pose.inverse())

    # take detected tag poses and combine them into a surface
    # neon_surface = Surface(mocap_surface.tag_size)
    neon_surface = Surface(130)
    for pose in neon_apriltags.tag_poses:
        neon_surface.add_tag_pose(pose)

    # build the surface from the tags
    ok = neon_surface.build_surface(
        orient_towards=neon.pose_in_tags[0].position, from_poses=True
    )
    if not ok:
        return None, None

    return neon_surface, neon_apriltags
