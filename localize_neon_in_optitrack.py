import argparse
import json

import cv2
import numpy as np

from apriltags import AprilTags
from load_optitrack_data import Take
from neon import Neon
from plots import (
    plot_apriltag_and_surface_in_neon,
    plot_neon_in_optitrack,
    plot_neon_in_surface,
    plot_surface_local_coordinate_system_in_optitrack,
)
from pose import Pose
from rigid import fit_plane
from surface import Surface

parser = argparse.ArgumentParser(
    description="Localize Neon in OptiTrack coordinate system using AprilTags."
)
parser.add_argument(
    "--image", required=True, help="Path to AprilTag image (e.g., frame_0089.png)"
)
parser.add_argument(
    "--calib",
    required=True,
    help="Path to scene camera calibration JSON (e.g., scene_camera.json)",
)
parser.add_argument(
    "--optitrack",
    required=True,
    help="Path to OptiTrack CSV data (e.g., NEON-DISPLAY CALIB Take ...csv)",
)
args = parser.parse_args()

img_path = args.image
calib_path = args.calib
optitrack_path = args.optitrack

# load apriltag image from Neon recording

img = cv2.imread(img_path)

# load scene camera calibration

scene_calib = []
with open(calib_path, "r") as f:
    scene_calib = json.load(f)

scene_camera_matrix = np.array(scene_calib["camera_matrix"])

K = scene_camera_matrix
D = np.array(scene_calib["distortion_coefficients"]).flatten()

neon = Neon(K, D)

# load optitrack data

d = Take()
d.readCSV(optitrack_path)

# extract neon's markers from optitrack data

neon_marker_positions_in_optitrack = np.zeros((3, 6))

# let's just work with a single frame from the middle of the take
# this could be modified to take use the average of all frames
for i in range(6):
    nframes = len(d.rigid_bodies["NEON_FRAME"].positions)
    neon_marker_positions_in_optitrack[:, i] = d.rigid_bodies[
        f"NEON_FRAME:Marker 00{i + 1}"
    ].positions[nframes // 2]

# extract display markers from optitrack data

display_positions_optitrack = np.zeros((4, 5, 3))

tc = 0
for i in range(20):
    mc = np.mod(i, 5)
    if mc == 0 and i > 4:
        tc += 1

    # let's just work with a single frame from the middle of the take
    # this could be modified to take use the average of all frames
    nframes = len(d.rigid_bodies["NEON_DISPLAY_CALIB"].positions)
    if i > 9:
        display_positions_optitrack[tc, mc, :] = d.rigid_bodies[
            f"NEON_DISPLAY_CALIB:Marker 0{i + 1}"
        ].positions[nframes // 2]
    else:
        display_positions_optitrack[tc, mc, :] = d.rigid_bodies[
            f"NEON_DISPLAY_CALIB:Marker 00{i + 1}"
        ].positions[nframes // 2]


# collect display marker positions in a single array
all_display_tag_markers_optitrack = display_positions_optitrack.reshape(-1, 3)

# extract neon's and display's rotations from optitrack data
# again, just work with a single frame from the middle of the take

# optitrack rotation matrices follow an unclear convention
# we instead estimate it ourselves below with SVD

# quat = d.rigid_bodies["NEON_FRAME"].rotations[nframes // 2]
# neon_rotation_opti = R.from_quat(quat).as_matrix()

# quat = d.rigid_bodies["NEON_DISPLAY_CALIB"].rotations[nframes // 2]
# display_rotation_opti = R.from_quat(quat).as_matrix()

# construct the estimated pose of the display in optitrack system
# we just calculate it ourselves
centroid, rotation = fit_plane(all_display_tag_markers_optitrack.T)
display_pose_optitrack = Pose(
    position=centroid.flatten(),
    rotation=rotation,
)

# estimate tag size [m] from optitrack data
tag_1_marker_positions_in_optitrack = all_display_tag_markers_optitrack[
    (all_display_tag_markers_optitrack[:, 2] > 1.27)
    & (all_display_tag_markers_optitrack[:, 0] < 0)
]
tag_size = np.sqrt(
    np.sum(
        np.power(
            tag_1_marker_positions_in_optitrack[1, :]
            - tag_1_marker_positions_in_optitrack[2, :],
            2,
        )
    )
)

# detect apriltags in neon image
apriltags = AprilTags(neon.camera_matrix, neon.dist_coeffs, tag_size)
apriltags.detect_tags(img)
apriltags.extract_tag_poses()

# find neon's pose in each apriltag coordinate system
for pose in apriltags.tag_poses:
    neon.add_pose_in_tag(pose.inverse())

# take detected tag poses and combine them into a surface
display_surface = Surface(tag_size)
for pose in apriltags.tag_poses:
    display_surface.add_tag_pose(pose)

# build the surface from the tags
display_surface.build_surface()

# find neon's pose in local surface coordinate system
# NOTE: the local surface coordinate system follows the conventions of our SVD method in `fit_plane`
# It does not follow the Optitrack rotation matrix conventions, but that is okay,
# as the end result is in the OptiTrack coordinate system.
neon.set_pose_in_surface(display_surface.pose_in_neon.inverse())

# apply surface pose in optitrack to neon pose in surface coordinates to get neon camera pose in optitrack coordinates
neon.calculate_pose_in_optitrack(display_pose_optitrack)

# plot tags and surface in neon camera coordinates as sanity check
plot_apriltag_and_surface_in_neon(
    apriltags,
    display_surface,
)

# plot neon's pose in display surface coordinates as sanity check
plot_neon_in_surface(
    neon.pose_in_surface,
    display_surface,
)

# plot surface local coordinate system in optitrack space,
# as obtained via SVD, as sanity check
plot_surface_local_coordinate_system_in_optitrack(
    all_display_tag_markers_optitrack,
    display_pose_optitrack,
)

cam_z_axis_in_optitrack = neon.pose_in_optitrack.rotation @ np.array([[0], [0], [1.0]])

# plot the final positions, as sanity check
plot_neon_in_optitrack(
    neon,
    all_display_tag_markers_optitrack,
    neon_marker_positions_in_optitrack,
    cam_z_axis_in_optitrack,
)

# Convert cam_z_axis_in_optitrack (a 3D vector) to spherical coordinates
# x: left (+), y: forward (+), z: up (+)
cam_vec = cam_z_axis_in_optitrack.flatten()
x, y, z = cam_vec
r = np.linalg.norm(cam_vec)
theta = np.arccos(z / r)  # inclination from z-axis
phi = np.arctan2(y, x)  # azimuth from x-axis (left)
theta = np.degrees(theta)  # convert to degrees
phi = np.degrees(phi)  # convert to degrees

print(
    f"Camera Z axis in spherical coordinates (r, theta, phi): ({r:.3f}, {theta:.3f}, {phi:.3f})"
)

print("\nAbsolute Neon scene camera pose in OptiTrack coordinates:\n")
print(neon.pose_in_optitrack)

# determine position of neon camera relative to frame markers
avg_neon_marker_positions = np.mean(neon_marker_positions_in_optitrack, axis=1)
neon_camera_position_relative_to_markers = (
    neon.pose_in_optitrack.position - avg_neon_marker_positions
)
neon_camera_pose_relative_to_markers = Pose(
    position=neon_camera_position_relative_to_markers,
    rotation=neon.pose_in_optitrack.rotation,
)

# invert ("recover") and plot neon scene camera relative to markers, as sanity check
neon_recovered = Neon(K, D)
neon_recovered.pose_in_optitrack = Pose(
    position=neon_camera_position_relative_to_markers + avg_neon_marker_positions,
    rotation=neon.pose_in_optitrack.rotation,
)
plot_neon_in_optitrack(
    neon_recovered,
    all_display_tag_markers_optitrack,
    neon_marker_positions_in_optitrack,
    cam_z_axis_in_optitrack,
)

print("\nNeon camera pose relative to frame markers (OptiTrack coordinates):\n")
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
