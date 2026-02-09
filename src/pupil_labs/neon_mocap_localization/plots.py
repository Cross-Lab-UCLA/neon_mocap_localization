from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from pupil_labs.neon_mocap_localization.apriltags import AprilTags
from pupil_labs.neon_mocap_localization.mocap import MocapHead, MocapSurface
from pupil_labs.neon_mocap_localization.neon import Neon
from pupil_labs.neon_mocap_localization.pose import Pose


def set_axes_equal(ax: Any) -> None:
    """Make axes of 3D plot have equal scale."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def plot_apriltags_in_neon(
    apriltags: AprilTags,
    tag_plane: npt.NDArray[np.float64],
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Transform tag corners to camera frame using tag_pose_neon
    tag_corners_in_cam = (
        apriltags.pose.rotation @ tag_plane.T
    ).T + apriltags.pose.position.T

    ax.plot(
        tag_corners_in_cam[:, 0],
        tag_corners_in_cam[:, 1],
        tag_corners_in_cam[:, 2],
        "g-",
        linewidth=2,
    )

    ax.quiver(
        *apriltags.pose.position,
        *apriltags.pose.rotation[:, 0],
        color="r",
        length=0.105,
        normalize=True,
    )
    ax.quiver(
        *apriltags.pose.position,
        *apriltags.pose.rotation[:, 1],
        color="g",
        length=0.105,
        normalize=True,
    )
    ax.quiver(
        *apriltags.pose.position,
        *apriltags.pose.rotation[:, 2],
        color="b",
        length=0.105,
        normalize=True,
    )

    # Plot camera origin
    ax.scatter(0, 0, 0, color="b", s=50, label="Neon Camera Origin")

    # Plot camera Z axis (forward direction)
    ax.quiver(
        0,
        0,
        0,
        0,
        0,
        0.2,
        color="r",
        length=0.205,
        normalize=True,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("AprilTag + Surface in Camera Coordinate System")
    ax.legend()

    set_axes_equal(ax)

    plt.show()


def plot_neon_in_surface(
    neon_pose_in_surface: Pose,
    best_plane: AprilTags,
    surface_points_3d: npt.NDArray[np.float64],
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        surface_points_3d[:, 0],
        surface_points_3d[:, 1],
        surface_points_3d[:, 2],
        "m-",
        linewidth=2,
        label="Virtual Surface",
    )

    # Plot camera origin
    ax.scatter(
        neon_pose_in_surface.position[0],
        neon_pose_in_surface.position[1],
        neon_pose_in_surface.position[2],
        color="b",
        s=50,
        label="Neon Camera Origin",
    )

    cam_z_axis_in_surface = neon_pose_in_surface.rotation @ np.array([
        [0],
        [0],
        [10.5],
    ])  # 50cm forward

    # Plot camera Z axis (forward direction)
    ax.quiver(
        neon_pose_in_surface.position[0],
        neon_pose_in_surface.position[1],
        neon_pose_in_surface.position[2],
        cam_z_axis_in_surface[0, 0],
        cam_z_axis_in_surface[1, 0],
        cam_z_axis_in_surface[2, 0],
        color="r",
        length=0.1,
        normalize=True,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Neon in Surface Coordinate System")

    ax.legend()
    set_axes_equal(ax)

    plt.show()


def plot_neon_in_mocap(
    neon: Neon,
    mocap_surface: MocapSurface,
    mocap_head: MocapHead,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if len(mocap_surface.apriltags):
        for ac, apriltag in enumerate(mocap_surface.apriltags):
            for mc, marker in enumerate(apriltag.markers):
                if ac == 0 and mc == 0:
                    ax.plot(
                        marker.Xs,
                        marker.Ys,
                        marker.Zs,
                        "ko",
                        label="Tag Markers",
                    )
                else:
                    ax.plot(
                        marker.Xs,
                        marker.Ys,
                        marker.Zs,
                        "ko",
                    )
    else:
        for mc, marker in enumerate(mocap_surface.markers):
            if mc == 0:
                ax.plot(
                    marker.Xs,
                    marker.Ys,
                    marker.Zs,
                    "ko",
                    label="Tag Markers",
                )
            else:
                ax.plot(
                    marker.Xs,
                    marker.Ys,
                    marker.Zs,
                    "ko",
                )

    for c, marker in enumerate(mocap_head.markers):
        if c == 0:
            ax.plot(
                marker.Xs,
                marker.Ys,
                marker.Zs,
                "rs",
                label="Neon Markers",
            )
        else:
            ax.plot(
                marker.Xs,
                marker.Ys,
                marker.Zs,
                "rs",
            )

    # plot local coord sys of mocap surface
    ax.quiver(
        *mocap_surface.pose.position,
        *mocap_surface.pose.rotation[:, 0],
        color="r",
        length=0.105,
        normalize=True,
    )
    ax.quiver(
        *mocap_surface.pose.position,
        *mocap_surface.pose.rotation[:, 1],
        color="g",
        length=0.105,
        normalize=True,
    )
    ax.quiver(
        *mocap_surface.pose.position,
        *mocap_surface.pose.rotation[:, 2],
        color="b",
        length=0.105,
        normalize=True,
    )

    # Plot estimated scene camera position
    ax.scatter(
        neon.reference_pose_in_mocap.position[0],
        neon.reference_pose_in_mocap.position[1],
        neon.reference_pose_in_mocap.position[2],
        color="b",
        marker="s",
        s=50,
        label="Estimated Neon Scene Camera Position",
    )

    # Plot camera Z axis (forward direction)
    ax.quiver(
        neon.reference_pose_in_mocap.position[0],
        neon.reference_pose_in_mocap.position[1],
        neon.reference_pose_in_mocap.position[2],
        neon.reference_camera_axis_in_mocap[0],
        neon.reference_camera_axis_in_mocap[1],
        neon.reference_camera_axis_in_mocap[2],
        color="r",
        length=0.1,
        normalize=True,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()
    ax.set_title("Neon in MoCap Coordinate System")
    set_axes_equal(ax)

    plt.show()


def plot_surface_local_coordinate_system_in_mocap(mocap_surface: MocapSurface) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if len(mocap_surface.apriltags) > 0:
        for apriltag in mocap_surface.apriltags:
            for marker in apriltag.markers:
                ax.scatter(
                    marker.Xs,
                    marker.Ys,
                    marker.Zs,
                    color="k",
                    s=50,
                )
    else:
        for marker in mocap_surface.markers:
            ax.scatter(
                marker.Xs,
                marker.Ys,
                marker.Zs,
                color="k",
                s=50,
            )

    ax.plot(
        mocap_surface.pose.position[0],
        mocap_surface.pose.position[1],
        mocap_surface.pose.position[2],
        "ro",
        label="MoCap Surface Origin",
    )

    ax.quiver(
        *mocap_surface.pose.position,
        *mocap_surface.pose.rotation[:, 0],
        color="r",
        length=0.105,
        normalize=True,
    )
    ax.quiver(
        *mocap_surface.pose.position,
        *mocap_surface.pose.rotation[:, 1],
        color="g",
        length=0.105,
        normalize=True,
    )
    ax.quiver(
        *mocap_surface.pose.position,
        *mocap_surface.pose.rotation[:, 2],
        color="b",
        length=0.105,
        normalize=True,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()
    ax.set_title("Local Surface Coordinate System")
    set_axes_equal(ax)

    plt.show()
