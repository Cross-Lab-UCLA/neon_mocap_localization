import cv2
import argparse
import json

import numpy as np
import pandas as pd
import pupil_labs.neon_recording as plnr
import matplotlib.pyplot as plt
from tqdm import tqdm

from cloud_recording import CloudRecording
from mocap import (
    MocapAprilTag,
    MocapHead,
    MocapIRMarker,
    MocapSurface,
    extract_apriltag_surface,
)
from neon import Neon
from plots import (
    plot_apriltag_and_surface_in_neon,
    plot_neon_in_mocap,
    plot_neon_in_surface,
    plot_surface_local_coordinate_system_in_mocap,
)
from pose import Pose
from surface import Surface
# import threed_utils

parser = argparse.ArgumentParser(
    description="Determines relative position of Neon scene camera in MoCap coordinate system"
)
parser.add_argument(
    "-r",
    "--neon_rec_path",
    help="The path to the Neon Recording Data",
    required=True,
)
parser.add_argument(
    "-m",
    "--mocap_path",
    help="The path to the MoCap data (CSV; in Neon timebase)",
    required=True,
)

args = vars(parser.parse_args())

# load data

# open neon recording and initialize neon object

neon_rec = None
is_cloud_rec = False
try:
    neon_rec = CloudRecording(args["neon_rec_path"])
    is_cloud_rec = True
    nframes = neon_rec.scene.nframes
except Exception:
    try:
        neon_rec = plnr.open(args["neon_rec_path"])
        nframes = len(neon_rec.scene.data)
    except Exception:
        raise ValueError("Not a valid Neon data directory")


neon = Neon(recording=neon_rec)

# load mocap data

marker_positions = pd.read_csv(args["mocap_path"])

print("Searching for most accurate localization...")
smallest_error = float("inf")

cv2.namedWindow("AprilTag Tracking", cv2.WINDOW_NORMAL)

last_ts = neon_rec.scene.time[-1]

rmses = []
for frame in tqdm(range(int(nframes))):
    neon_timestamp = neon_rec.scene.time[frame]

    if is_cloud_rec:
        apriltag_img = neon_rec.scene.bgr_at_time(neon_timestamp)
    else:
        apriltag_img = neon_rec.scene.data[frame].bgr

    if apriltag_img is None:
        continue

    # 2. Display the resulting frame
    cv2.imshow("AprilTag Tracking", apriltag_img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # find the equivalent marker positions based on neon timestamp
    if "timestamp [ns]" in marker_positions:
        diffs = (marker_positions["timestamp [ns]"] - neon_timestamp).abs()
        markers_for_calib = marker_positions.iloc[diffs.idxmin()]
    else:
        markers_for_calib = marker_positions.iloc[len(marker_positions) // 2]

    if np.isnan(markers_for_calib["T1TL_X"]).any():
        continue

    # holds the mocap surface data for a collection of AprilTags
    mocap_surface = MocapSurface()

    for tag_id, tag_num in enumerate(["1", "2", "3", "4"]):
        mocap_apriltag = MocapAprilTag(tag_id)

        for tag_corner in ["TL", "TR", "BR", "BL"]:
            marker_pos_X = markers_for_calib[f"T{tag_num}{tag_corner}_X"].squeeze()
            marker_pos_Y = markers_for_calib[f"T{tag_num}{tag_corner}_Y"].squeeze()
            marker_pos_Z = markers_for_calib[f"T{tag_num}{tag_corner}_Z"].squeeze()

            mocap_apriltag.add_marker(
                MocapIRMarker(
                    marker_pos_X,
                    marker_pos_Y,
                    marker_pos_Z,
                )
            )

        mocap_apriltag.estimate_tag_center()
        mocap_surface.add_apriltag(mocap_apriltag)

    # extract the marker positions for the head pose into a convenient object
    mocap_head = MocapHead()

    neon_marker_num = 1
    while True:
        marker_name = f"NEON_MARKER_{neon_marker_num}"
        if marker_name + "_X" not in markers_for_calib.keys():
            break

        marker_pos_X = markers_for_calib[f"{marker_name}_X"].squeeze()
        marker_pos_Y = markers_for_calib[f"{marker_name}_Y"].squeeze()
        marker_pos_Z = markers_for_calib[f"{marker_name}_Z"].squeeze()

        mocap_head.add_marker(
            MocapIRMarker(
                marker_pos_X,
                marker_pos_Y,
                marker_pos_Z,
            )
        )

        neon_marker_num += 1

    # now that we have collected the apriltags into a surface object,
    # we can construct its pose in neon scene cam coordinates
    head_marker_pos = np.array(
        [mocap_head.markers[0].Xs, mocap_head.markers[0].Ys, mocap_head.markers[0].Zs]
    )
    ok = mocap_surface.construct_pose(orient_towards=head_marker_pos)
    if not ok:
        continue

    neon_surface, neon_apriltags = extract_apriltag_surface(
        neon, mocap_surface, apriltag_img
    )
    if neon_surface is None or neon_apriltags is None:
        continue

    # find pose pair with smallest tag pose error
    for idx, error in enumerate(neon_apriltags.reprojection_errors):
        if error < smallest_error:
            smallest_error = error
            best_apriltag_corners = neon_apriltags.tag_corners[idx]

            for apriltag in mocap_surface.apriltags:
                if apriltag.tag_id == neon_apriltags.tag_ids[idx]:
                    best_mocap_markers = np.array(
                        [
                            [
                                [marker.Xs, marker.Ys, marker.Zs]
                                for marker in apriltag.markers
                            ]
                        ]
                    ).squeeze()
                    best_apriltag_idx = idx
                    break
        else:
            continue

    tag_surface = Surface(130)
    for corner in best_apriltag_corners:
        tag_surface.surface_corners.append(corner)

    ok = tag_surface.build_surface(
        orient_towards=neon.pose_in_tags[0].position, from_poses=False
    )
    if not ok:
        continue

    mocap_tag_surface = MocapSurface()
    mocap_tag_surface.add_apriltag(mocap_surface.apriltags[best_apriltag_idx])

    centroid = np.mean(best_mocap_markers, axis=0)

    mocap_tag_surface.x_axis = best_mocap_markers[1] - best_mocap_markers[0]
    mocap_tag_surface.x_axis /= np.linalg.norm(mocap_tag_surface.x_axis)

    mocap_tag_surface.y_axis = best_mocap_markers[3] - best_mocap_markers[0]
    mocap_tag_surface.y_axis /= np.linalg.norm(mocap_tag_surface.y_axis)

    mocap_tag_surface.normal = np.cross(
        mocap_tag_surface.x_axis, mocap_tag_surface.y_axis
    )
    mocap_tag_surface.normal /= np.linalg.norm(mocap_tag_surface.normal)

    orient_towards = head_marker_pos
    if orient_towards is not None:
        orient_towards = np.asarray(orient_towards, dtype=float)

        ref_vec = orient_towards - centroid.squeeze()

        if np.dot(mocap_tag_surface.normal, ref_vec) < 0:
            mocap_tag_surface.normal = -mocap_tag_surface.normal

    rotation = np.zeros((3, 3))
    rotation[:, 0] = mocap_tag_surface.x_axis
    rotation[:, 1] = mocap_tag_surface.y_axis
    rotation[:, 2] = mocap_tag_surface.normal

    mocap_tag_surface.pose = Pose(
        position=centroid.flatten(),
        rotation=rotation,
    )

    # apply calibration data

    # find neon's pose in local surface coordinate system
    # NOTE: the local surface coordinate system follows the conventions of
    # our SVD method in `fit_plane`.
    # neon.set_pose_in_surface(neon_surface.pose_in_neon.inverse())
    neon.set_pose_in_surface(tag_surface.pose_in_neon.inverse())

    # apply surface pose in mocap sys to neon pose in surface coordinates
    # to get neon camera pose in mocap coordinates
    # neon.calculate_pose_in_mocap(mocap_surface.pose)
    neon.calculate_pose_in_mocap(mocap_tag_surface.pose)

    # neon_pos_mocap, neon_rot_mocap, rmse = neon.calculate_pose_in_mocap(
    #     mocap_surface,
    #     best_mocap_markers,
    #     best_apriltag_corners,
    #     neon_apriltags.new_K,
    #     neon_apriltags.D,
    # )
    # if neon_pos_mocap is None or neon_rot_mocap is None or rmse is None:
    #     continue

    # rmses.append(rmse)

    # neon.pose_in_mocap = Pose(
    # position=neon_pos_mocap,
    # rotation=neon_rot_mocap,
    # )

    # determine position of neon camera relative to frame markers
    try:
        neon_marker_positions_in_mocap = np.array(
            [
                [ir_marker.Xs, ir_marker.Ys, ir_marker.Zs]
                for ir_marker in mocap_head.markers
            ]
        ).T
        avg_neon_marker_positions = np.mean(neon_marker_positions_in_mocap, axis=1)
        neon_camera_position_relative_to_markers = (
            neon.pose_in_mocap.position - avg_neon_marker_positions
        )
    except Exception as e:
        print(e)
        continue

    neon_camera_pose_relative_to_markers = Pose(
        position=neon_camera_position_relative_to_markers,
        rotation=neon.pose_in_mocap.rotation,
    )

    best_calib_data = {
        "best_frame": frame,
        "timestamp": neon_timestamp,
        "neon_surface": neon_surface,
        "neon_apriltags": neon_apriltags,
        "neon_camera_pose_relative_to_markers": neon_camera_pose_relative_to_markers,
        "mocap_surface": mocap_surface,
        "mocap_head": mocap_head,
        "rmses": rmses,
    }

cv2.destroyAllWindows()

plt.hist(rmses)
plt.show()

# plot tags and surface in neon camera coordinates as sanity check
plot_apriltag_and_surface_in_neon(
    best_calib_data["neon_apriltags"],
    best_calib_data["neon_surface"],
)

# plot neon's pose in display surface coordinates as sanity check
plot_neon_in_surface(
    neon.pose_in_surface,
    best_calib_data["neon_surface"],
)

# plot surface local coordinate system in mocap space,
# as obtained via SVD, as sanity check
plot_surface_local_coordinate_system_in_mocap(best_calib_data["mocap_surface"])

cam_z_axis_in_mocap = neon.pose_in_mocap.rotation @ np.array([[0], [0], [1.0]])

# plot the final positions, as sanity check
plot_neon_in_mocap(
    neon,
    best_calib_data["mocap_surface"],
    best_calib_data["mocap_head"],
    cam_z_axis_in_mocap,
)

print("\nAbsolute Neon scene camera pose in MoCap coordinates:\n")
print(neon.pose_in_mocap)

# invert ("recover") and plot neon scene camera relative to markers, as sanity check
neon_recovered = Neon(recording=neon_rec)
neon_recovered.pose_in_mocap = Pose(
    position=(
        best_calib_data["neon_camera_pose_relative_to_markers"].position
        + avg_neon_marker_positions
    ),
    rotation=neon.pose_in_mocap.rotation,
)
plot_neon_in_mocap(
    neon_recovered,
    best_calib_data["mocap_surface"],
    best_calib_data["mocap_head"],
    cam_z_axis_in_mocap,
)

# Export neon_camera_pose_relative_to_markers to JSON file
output = {
    "position": best_calib_data[
        "neon_camera_pose_relative_to_markers"
    ].position.tolist(),
    "rotation": best_calib_data[
        "neon_camera_pose_relative_to_markers"
    ].rotation.tolist(),
}

with open("neon_camera_pose_relative_to_markers.json", "w") as f:
    json.dump(output, f, indent=4)

print(
    "\nExported neon_camera_pose_relative_to_markers to neon_camera_pose_relative_to_markers.json"
)

# make new columns in marker_positions.csv for:
# - gaze origin in mocap coord sys at each frame
# - gaze direction in mocap coord sys at each frame

# gaze_origin_Xs = np.zeros(len(marker_positions))
# gaze_origin_Ys = np.zeros(len(marker_positions))
# gaze_origin_Zs = np.zeros(len(marker_positions))

# gaze_dir_Xs = np.zeros(len(marker_positions))
# gaze_dir_Ys = np.zeros(len(marker_positions))
# gaze_dir_Zs = np.zeros(len(marker_positions))

# for frame in tqdm(range(len(marker_positions))):
#     marker_timestamp = marker_positions["timestamp [ns]"].iloc[frame]

#     # find the equivalent neon data based on marker timestamp
#     idx = np.searchsorted(neon_rec.gaze.time, marker_timestamp)

#     gaze_x = neon_rec.gaze.data["point_x"][idx]
#     gaze_y = neon_rec.gaze.data["point_y"][idx]

#     gaze_dir = threed_utils.unproject_points(
#         np.array([gaze_x, gaze_y]),
#         neon_rec.calibration.scene_camera_matrix,
#         neon_rec.calibration.scene_distortion_coefficients,
#         normalize=True,
#     )

#     gaze_dir_mocap = (R_neon_to_mocap @ gaze_dir.reshape(3, -1)).squeeze()
#     gaze_dir_Xs[frame] = gaze_dir_mocap[0]
#     gaze_dir_Ys[frame] = gaze_dir_mocap[1]
#     gaze_dir_Zs[frame] = gaze_dir_mocap[2]

#     markers_for_calib = marker_positions.iloc[frame]

#     neon_marker_num = 1
#     marker_pos_X = []
#     marker_pos_Y = []
#     marker_pos_Z = []
#     while True:
#         marker_name = f"NEON_MARKER_{neon_marker_num}"
#         if marker_name + "_X" not in markers_for_calib.keys():
#             break

#         marker_pos_X.append(markers_for_calib[f"{marker_name}_X"].squeeze())
#         marker_pos_Y.append(markers_for_calib[f"{marker_name}_Y"].squeeze())
#         marker_pos_Z.append(markers_for_calib[f"{marker_name}_Z"].squeeze())

#         neon_marker_num += 1

#     avg_neon_marker_position = np.array(
#         [
#             np.mean(marker_pos_X),
#             np.mean(marker_pos_Y),
#             np.mean(marker_pos_Z),
#         ]
#     )

#     gaze_origin_mocap = (
#         neon_camera_position_relative_to_markers + avg_neon_marker_position
#     )
#     gaze_origin_Xs[frame] = gaze_origin_mocap[0]
#     gaze_origin_Ys[frame] = gaze_origin_mocap[1]
#     gaze_origin_Zs[frame] = gaze_origin_mocap[2]

# marker_positions["gaze_origin_X"] = gaze_origin_Xs
# marker_positions["gaze_origin_Y"] = gaze_origin_Ys
# marker_positions["gaze_origin_Z"] = gaze_origin_Zs

# marker_positions["gaze_dir_X"] = gaze_dir_Xs
# marker_positions["gaze_dir_Y"] = gaze_dir_Ys
# marker_positions["gaze_dir_Z"] = gaze_dir_Zs

# marker_positions.to_csv("marker_positions_w_gaze.csv")
