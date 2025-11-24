import argparse
import json

import numpy as np
import pandas as pd
import pupil_labs.neon_recording as plnr

from mocap import (
    MocapAprilTag,
    MocapCalibData,
    MocapHead,
    MocapIRMarker,
    MocapSurface,
    unify_calib_data,
)
from neon import Neon
from plots import (
    plot_apriltag_and_surface_in_neon,
    plot_neon_in_mocap,
    plot_neon_in_surface,
    plot_surface_local_coordinate_system_in_mocap,
)
from pose import Pose

# parser = argparse.ArgumentParser(
#     description="Determines relative position of Neon scene camera in MoCap coordinate system"
# )
# parser.add_argument(
#     "-r",
#     "--neon_rec_path",
#     help="The path to the Neon Native Recording Data",
#     required=True,
# )
# parser.add_argument(
#     "-m",
#     "--mocap_path",
#     help="The path to the MoCap data (CSV; in Neon timebase)",
#     required=True,
# )

# args = vars(parser.parse_args())

# # load data

# neon_rec = plnr.open(args["neon_rec_path"])
# marker_positions = pd.read_csv(args["mocap_path"])

# debug data

neon_rec = plnr.open(
    "Static_Test-20251028T211908Z-1-001/Static_Test/Native Recording Data/static_test_2025-10-14_12-16-38-eae4a97c/"
)
marker_positions = pd.read_csv("marker_positions.csv")

# get an apriltag image from Neon recording
# for now, we just test with a single image

nframes = len(neon_rec.scene.data)
apriltag_img = neon_rec.scene.data[nframes // 2 + 100].bgr
neon_timestamp = neon_rec.scene.time[nframes // 2 + 100]

# initialize neon object with camera calibration data

neon = Neon(recording=neon_rec)

# matrix that converts between coordinate systems of Neon and MoCap
# (when following our recommendations in README.md)
# y and z are swapped and vertical is reversed
R_apriltag_to_mocap = np.array(
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ]
)


# find the equivalent marker positions based on neon timestamp
diffs = (marker_positions["timestamps [ns]"] - neon_timestamp).abs()
markers_for_calib = marker_positions.iloc[diffs.idxmin()]

# holds the mocap surface data for a collection of AprilTags
mocap_surface = MocapSurface()

for tag_num in ["1", "2", "3"]:  # , "4"]:
    mocap_apriltag = MocapAprilTag()

    for tag_corner in ["TL", "TR", "BL", "BR"]:
        # index = marker_indices[f"T{tag_num}{tag_corner}"]
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

    mocap_surface.add_apriltag(mocap_apriltag)

# now that we have collected the apriltags into a surface object,
# we can construct its pose in neon scene cam coordinates
mocap_surface.construct_pose()

# extract the marker positions for the head pose into a convenient object
mocap_head = MocapHead()

# neon_markers = [
# key for key in markers_for_calib.keys() if key.startswith("NEON_MARKER_")
# ]
# for marker in neon_markers:
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

mocap_calib_data = MocapCalibData(mocap_surface, mocap_head)
neon_surface, neon_apriltags = unify_calib_data(
    neon, mocap_surface, apriltag_img, R_apriltag_to_mocap
)

# plot tags and surface in neon camera coordinates as sanity check
plot_apriltag_and_surface_in_neon(
    neon_apriltags,
    neon_surface,
)

# plot neon's pose in display surface coordinates as sanity check
plot_neon_in_surface(
    neon.pose_in_surface,
    neon_surface,
)

# plot surface local coordinate system in mocap space,
# as obtained via SVD, as sanity check
plot_surface_local_coordinate_system_in_mocap(mocap_surface)

cam_z_axis_in_mocap = neon.pose_in_mocap.rotation @ np.array([[0], [0], [1.0]])

# plot the final positions, as sanity check
plot_neon_in_mocap(
    neon,
    mocap_surface,
    mocap_head,
    cam_z_axis_in_mocap,
)

# Convert cam_z_axis_in_mocap (a 3D vector) to spherical coordinates
# x: left (+), y: forward (+), z: up (+)
cam_vec = cam_z_axis_in_mocap.flatten()
x, y, z = cam_vec

r = np.linalg.norm(cam_vec)
theta = np.degrees(np.arccos(z / r))  # inclination from z-axis
phi = np.degrees(np.arctan2(y, x))  # azimuth from x-axis (left)

print(
    f"Camera Z axis in spherical coordinates (r, theta, phi): ({r:.3f}, {theta:.3f}, {phi:.3f})"
)

print("\nAbsolute Neon scene camera pose in MoCap coordinates:\n")
print(neon.pose_in_mocap)

# determine position of neon camera relative to frame markers
neon_marker_positions_in_mocap = np.array(
    [[ir_marker.Xs, ir_marker.Ys, ir_marker.Zs] for ir_marker in mocap_head.markers]
).T
avg_neon_marker_positions = np.mean(neon_marker_positions_in_mocap, axis=1)
neon_camera_position_relative_to_markers = (
    neon.pose_in_mocap.position - avg_neon_marker_positions
)
neon_camera_pose_relative_to_markers = Pose(
    position=neon_camera_position_relative_to_markers,
    rotation=neon.pose_in_mocap.rotation,
)

# invert ("recover") and plot neon scene camera relative to markers, as sanity check
neon_recovered = Neon(recording=neon_rec)
neon_recovered.pose_in_mocap = Pose(
    position=neon_camera_position_relative_to_markers + avg_neon_marker_positions,
    rotation=neon.pose_in_mocap.rotation,
)
plot_neon_in_mocap(
    neon_recovered,
    mocap_surface,
    mocap_head,
    cam_z_axis_in_mocap,
)

print("\nNeon camera pose relative to frame markers (Mocap coordinates):\n")
print(neon_camera_pose_relative_to_markers)

# Export neon_camera_pose_relative_to_markers to JSON file
output = {
    "position": neon_camera_pose_relative_to_markers.position.tolist(),
    "rotation": neon_camera_pose_relative_to_markers.rotation.tolist(),
}

with open("neon_camera_pose_relative_to_markers.json", "w") as f:
    json.dump(output, f, indent=4)

print(
    "\nExported neon_camera_pose_relative_to_markers to neon_camera_pose_relative_to_markers.json"
)
