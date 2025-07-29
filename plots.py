import matplotlib.pyplot as plt
import numpy as np


def plot_apriltag_and_surface_in_neon(
    apriltags,
    display_surface,
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
        ax.quiver(*origin, *axes[:, 0], color="r", length=0.05, normalize=True)
        ax.quiver(*origin, *axes[:, 1], color="g", length=0.05, normalize=True)
        ax.quiver(*origin, *axes[:, 2], color="b", length=0.05, normalize=True)

    surface_corners = display_surface.surface_corners.copy()
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
        length=0.05,
        normalize=True,
    )

    ax.quiver(
        *(display_surface.pose_in_neon.position),
        *(display_surface.x_axis),
        color="r",
        length=0.05,
        normalize=True,
    )
    ax.quiver(
        *(display_surface.pose_in_neon.position),
        *(display_surface.y_axis),
        color="g",
        length=0.05,
        normalize=True,
    )
    ax.quiver(
        *(display_surface.pose_in_neon.position),
        *(display_surface.normal),
        color="b",
        length=0.05,
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
    display_surface,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surface_plane = np.zeros((5, 3))
    for i, s in enumerate(display_surface.surface_corners):
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
        [[0], [0], [0.5]]
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


def plot_neon_in_optitrack(
    neon,
    all_display_tag_markers_optitrack,
    neon_marker_positions_in_optitrack,
    cam_z_axis_in_optitrack,
):
    tag_1_marker_positions_in_optitrack = all_display_tag_markers_optitrack[
        (all_display_tag_markers_optitrack[:, 2] > 1.27)
        & (all_display_tag_markers_optitrack[:, 0] < 0)
    ]
    tag_2_marker_positions_in_optitrack = all_display_tag_markers_optitrack[
        (all_display_tag_markers_optitrack[:, 2] > 1.27)
        & (all_display_tag_markers_optitrack[:, 0] > 0)
    ]
    tag_3_marker_positions_in_optitrack = all_display_tag_markers_optitrack[
        (all_display_tag_markers_optitrack[:, 2] < 1.27)
        & (all_display_tag_markers_optitrack[:, 0] > 0)
    ]
    tag_4_marker_positions_in_optitrack = all_display_tag_markers_optitrack[
        (all_display_tag_markers_optitrack[:, 2] < 1.27)
        & (all_display_tag_markers_optitrack[:, 0] < 0)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        tag_1_marker_positions_in_optitrack[:, 0],
        tag_1_marker_positions_in_optitrack[:, 1],
        tag_1_marker_positions_in_optitrack[:, 2],
        "ko",
        label="Tag Markers",
    )
    ax.plot(
        tag_1_marker_positions_in_optitrack[0, 0],
        tag_1_marker_positions_in_optitrack[0, 1],
        tag_1_marker_positions_in_optitrack[0, 2],
        "go",
    )

    ax.plot(
        tag_2_marker_positions_in_optitrack[:, 0],
        tag_2_marker_positions_in_optitrack[:, 1],
        tag_2_marker_positions_in_optitrack[:, 2],
        "ko",
    )
    ax.plot(
        tag_2_marker_positions_in_optitrack[1, 0],
        tag_2_marker_positions_in_optitrack[1, 1],
        tag_2_marker_positions_in_optitrack[1, 2],
        "ro",
    )

    ax.plot(
        tag_3_marker_positions_in_optitrack[:, 0],
        tag_3_marker_positions_in_optitrack[:, 1],
        tag_3_marker_positions_in_optitrack[:, 2],
        "ko",
    )
    ax.plot(
        tag_3_marker_positions_in_optitrack[3, 0],
        tag_3_marker_positions_in_optitrack[3, 1],
        tag_3_marker_positions_in_optitrack[3, 2],
        "bo",
    )

    ax.plot(
        tag_4_marker_positions_in_optitrack[:, 0],
        tag_4_marker_positions_in_optitrack[:, 1],
        tag_4_marker_positions_in_optitrack[:, 2],
        "ko",
    )
    ax.plot(
        tag_4_marker_positions_in_optitrack[1, 0],
        tag_4_marker_positions_in_optitrack[1, 1],
        tag_4_marker_positions_in_optitrack[1, 2],
        "mo",
    )

    ax.plot(
        neon_marker_positions_in_optitrack[0, :],
        neon_marker_positions_in_optitrack[1, :],
        neon_marker_positions_in_optitrack[2, :],
        "rs",
        label="Neon Markers",
    )

    # Plot estimated scene camera position
    ax.scatter(
        neon.pose_in_optitrack.position[0],
        neon.pose_in_optitrack.position[1],
        neon.pose_in_optitrack.position[2],
        color="b",
        marker="s",
        s=50,
        label="Estimated Neon Scene Camera Position",
    )

    # Plot camera Z axis (forward direction)
    ax.quiver(
        neon.pose_in_optitrack.position[0],
        neon.pose_in_optitrack.position[1],
        neon.pose_in_optitrack.position[2],
        cam_z_axis_in_optitrack[0, 0],
        cam_z_axis_in_optitrack[1, 0],
        cam_z_axis_in_optitrack[2, 0],
        color="r",
        length=0.1,
        normalize=True,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()
    ax.set_title("Neon in OptiTrack Coordinate System")
    ax.set_box_aspect([1, 1, 0.5])

    plt.show()


def plot_surface_local_coordinate_system_in_optitrack(
    all_display_tag_markers_optitrack,
    display_pose_optitrack,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for marker in all_display_tag_markers_optitrack:
        ax.scatter(
            marker[0],
            marker[1],
            marker[2],
            color="k",
            s=50,
        )

    ax.plot(
        display_pose_optitrack.position[0],
        display_pose_optitrack.position[1],
        display_pose_optitrack.position[2],
        "ro",
        label="Display Surface Origin",
    )

    ax.quiver(
        *display_pose_optitrack.position,
        *display_pose_optitrack.rotation[:, 0],
        color="r",
        length=0.05,
        normalize=True,
    )
    ax.quiver(
        *display_pose_optitrack.position,
        *display_pose_optitrack.rotation[:, 1],
        color="g",
        length=0.05,
        normalize=True,
    )
    ax.quiver(
        *display_pose_optitrack.position,
        *display_pose_optitrack.rotation[:, 2],
        color="b",
        length=0.05,
        normalize=True,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()
    ax.set_title("Local Display Surface Coordinate System")
    ax.set_box_aspect([1, 1, 0.5])

    plt.show()
