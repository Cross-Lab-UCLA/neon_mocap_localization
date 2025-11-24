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

    def construct_pose(self):
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

        poses = np.vstack([xs, ys, zs])
        print(poses.shape)

        centroid, rotation = fit_plane(poses)
        self.pose = Pose(
            position=centroid.flatten(),
            rotation=rotation,
        )

        apriltag.estimate_size_mm()
        self.tag_size = apriltag.tag_size


class MocapCalibData(object):
    """
    Holds the IR Maker Mocap calibration data.
    """

    def __init__(self, surface, head):
        self.surface = surface
        self.head = head


def unify_calib_data(neon, mocap_surface, img, R_apriltag_to_mocap):
    # detect apriltags in neon image
    neon_apriltags = AprilTags(neon, mocap_surface.tag_size, img)

    # find neon's pose in each apriltag coordinate system
    for pose in neon_apriltags.tag_poses:
        neon.add_pose_in_tag(pose.inverse())

    # take detected tag poses and combine them into a surface
    neon_surface = Surface(mocap_surface.tag_size)
    for pose in neon_apriltags.tag_poses:
        neon_surface.add_tag_pose(pose)

    # build the surface from the tags
    neon_surface.build_surface(
        orient_towards=neon.pose_in_tags[0].position, from_poses=False
    )

    # find neon's pose in local surface coordinate system
    # NOTE: the local surface coordinate system follows the conventions of
    # our SVD method in `fit_plane`.
    # It does not follow the MoCap coordinate system conventions, but that is okay,
    # as the end result is in the MoCap coordinate system.
    neon.set_pose_in_surface(neon_surface.pose_in_neon.inverse())

    # apply surface pose in mocap sys to neon pose in surface coordinates
    # to get neon camera pose in mocap coordinates
    neon.calculate_pose_in_mocap(mocap_surface.pose, R_apriltag_to_mocap)

    return neon_surface, neon_apriltags
