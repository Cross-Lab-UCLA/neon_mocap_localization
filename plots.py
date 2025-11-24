import matplotlib.pyplot as plt
import numpy as np


def plot_apriltag_and_surface_in_neon(
    apriltags,
    neon_surface,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # AprilTag corners in local tag frame (XY plane, Z=0)
    tag_half = apriltags.tag_size / 2
    tag_plane = np.array(
        [
            [-tag_half, -tag_half, 0],
            [tag_half, -tag_half, 0],
            [tag_half, tag_half, 0],
            [-tag_half, tag_half, 0],
            [-tag_half, -tag_half, 0],
        ]
    )

    for i, tag_pose_neon in enumerate(apriltags.tag_poses):
        # Transform tag corners to camera frame using tag_pose_neon
        tag_corners_in_cam = (
            tag_pose_neon.rotation @ tag_plane.T
        ).T + tag_pose_neon.position

        if i == 0:
            ax.plot(
                tag_corners_in_cam[:, 0],
                tag_corners_in_cam[:, 1],
                tag_corners_in_cam[:, 2],
                "g-",
                linewidth=2,
                label="AprilTag",
            )
        else:
            ax.plot(
                tag_corners_in_cam[:, 0],
                tag_corners_in_cam[:, 1],
                tag_corners_in_cam[:, 2],
                "g-",
                linewidth=2,
            )

        # Plot tag center
        ax.scatter(
            tag_pose_neon.position[0],
            tag_pose_neon.position[1],
            tag_pose_neon.position[2],
            color="g",
            s=50,
        )

        # Draw tag axes
        origin = tag_pose_neon.position
        axes = tag_pose_neon.rotation
        ax.quiver(*origin, *axes[:, 0], color="r", length=200.05, normalize=True)
        ax.quiver(*origin, *axes[:, 1], color="g", length=200.05, normalize=True)
        ax.quiver(*origin, *axes[:, 2], color="b", length=200.05, normalize=True)

    surface_corners = neon_surface.surface_corners.copy()
    surface_corners.append(surface_corners[0])  # Close the loop
    surface_corners_to_plot = np.array(surface_corners)
    ax.plot(
        surface_corners_to_plot[:, 0],
        surface_corners_to_plot[:, 1],
        surface_corners_to_plot[:, 2],
        color="m",
        linewidth=2,
        label="Surface",
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
        length=200.05,
        normalize=True,
    )

    ax.quiver(
        *(neon_surface.pose_in_neon.position),
        *(neon_surface.x_axis),
        color="r",
        length=300.05,
        normalize=True,
    )
    ax.quiver(
        *(neon_surface.pose_in_neon.position),
        *(neon_surface.y_axis),
        color="g",
        length=300.05,
        normalize=True,
    )
    ax.quiver(
        *(neon_surface.pose_in_neon.position),
        *(neon_surface.normal),
        color="b",
        length=300.05,
        normalize=True,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("AprilTag + Surface in Camera Coordinate System")
    ax.legend()

    ax.set_box_aspect([1, 1, 0.5])

    plt.show()


def plot_neon_in_surface(
    neon_pose_in_surface,
    neon_surface,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surface_plane = np.zeros((len(neon_surface.surface_corners), 3))
    for i, s in enumerate(neon_surface.surface_corners):
        surface_plane[i] = (
            neon_pose_in_surface.rotation @ s + neon_pose_in_surface.position
        )

    surface_plane[-1] = surface_plane[0]  # Close the loop

    ax.plot(
        surface_plane[:, 0],
        surface_plane[:, 1],
        surface_plane[:, 2],
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

    cam_z_axis_in_surface = neon_pose_in_surface.rotation @ np.array(
        [[0], [0], [10.5]]
    )  # 50cm forward

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
    ax.set_box_aspect([1, 1, 0.5])

    plt.show()


def plot_neon_in_mocap(
    neon,
    mocap_surface,
    mocap_head,
    cam_z_axis_in_mocap,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

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

    # Plot estimated scene camera position
    ax.scatter(
        neon.pose_in_mocap.position[0],
        neon.pose_in_mocap.position[1],
        neon.pose_in_mocap.position[2],
        color="b",
        marker="s",
        s=50,
        label="Estimated Neon Scene Camera Position",
    )

    # Plot camera Z axis (forward direction)
    ax.quiver(
        neon.pose_in_mocap.position[0],
        neon.pose_in_mocap.position[1],
        neon.pose_in_mocap.position[2],
        cam_z_axis_in_mocap[0, 0],
        cam_z_axis_in_mocap[1, 0],
        cam_z_axis_in_mocap[2, 0],
        color="r",
        length=0.1,
        normalize=True,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()
    ax.set_title("Neon in MoCap Coordinate System")
    ax.set_box_aspect([1, 1, 0.5])

    plt.show()


def plot_surface_local_coordinate_system_in_mocap(mocap_surface):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for apriltag in mocap_surface.apriltags:
        for marker in apriltag.markers:
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
        length=30.05,
        normalize=True,
    )
    ax.quiver(
        *mocap_surface.pose.position,
        *mocap_surface.pose.rotation[:, 1],
        color="g",
        length=15.05,
        normalize=True,
    )
    ax.quiver(
        *mocap_surface.pose.position,
        *mocap_surface.pose.rotation[:, 2],
        color="b",
        length=15.05,
        normalize=True,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()
    ax.set_title("Local Surface Coordinate System")
    ax.set_box_aspect([1, 1, 0.5])

    plt.show()
