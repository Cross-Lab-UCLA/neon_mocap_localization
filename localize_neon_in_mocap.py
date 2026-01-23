import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pupil_labs.neon_recording as plnr
from pupil_apriltags import Detector
from tqdm import tqdm

import threed_utils
from apriltags import AprilTags
from cloud_recording import CloudRecording
from mocap import MocapAprilTag, MocapHead, MocapIRMarker, MocapSurface
from neon import Neon
from plots import (plot_apriltags_in_neon, plot_neon_in_mocap,
                   plot_neon_in_surface,
                   plot_surface_local_coordinate_system_in_mocap)
from plots import (
    plot_apriltags_in_neon,
    plot_neon_in_mocap,
    plot_neon_in_surface,
    plot_surface_local_coordinate_system_in_mocap,
)
from pose import Pose

# from surface import Surface

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
# load config

config = []
with open(args["config_path"], "r") as f:
    config = json.load(f)

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

# load mocap data

marker_positions = pd.read_csv(args["mocap_path"])

# load apriltag corner data (in a user-specified local coordinate system)

# tags should be listed in order of increasing ID
# should be tag 0, tag 1, tag 2, tag 3
# order for corners of each tag should be -> BL BR TR TL
# x, y only
# units should be meters
tag_corner_coordinates = config["apriltag_corner_local_coordinates"]
plane_width = 0
plane_height = 0
for k, v in tag_corner_coordinates.items():
    m = np.zeros((len(v), 3), dtype=np.float32)
    v = np.array(v) * config["corner_unit_conversion_factor"]

    m[:, :2] = v

    tag_corner_coordinates[k] = m

    if np.max(v[:, 0]) > plane_width:
        plane_width = np.max(v[:, 0])

    if np.max(v[:, 1]) > plane_height:
        plane_height = np.max(v[:, 1])

for k, v in tag_corner_coordinates.items():
    v[:, 0] -= plane_width / 2
    v[:, 1] -= plane_height / 2

    for c in range(len(tag_corner_coordinates[k])):
        tag_corner_coordinates[k][c] = np.array(config["T_neon_to_mocap"]) @ v[c]

plane_points_3d = np.array(
    [
        [-plane_width / 2, plane_height / 2, 0],  # BL
        [plane_width / 2, plane_height / 2, 0],  # BR
        [plane_width / 2, -plane_height / 2, 0],  # TR
        [-plane_width / 2, -plane_height / 2, 0],  # TL
        [-plane_width / 2, plane_height / 2, 0],  # BL
    ]
)
for c in range(len(plane_points_3d)):
    plane_points_3d[c] = np.array(config["T_neon_to_mocap"]) @ plane_points_3d[c]

neon = Neon(recording=neon_rec)
apriltag_detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)

# first find best apriltag detection
print("Searching for most accurate localization...")
best_calib_data = {}
smallest_error = np.inf
best_timestamp = 0
best_frame = 0
best_plane = None
for frame in tqdm(range(int(nframes))):
    neon_timestamp = neon_rec.scene.time[frame]

    # find the equivalent marker positions based on neon timestamp
    if "timestamp [ns]" in marker_positions:
        diffs = (marker_positions["timestamp [ns]"] - neon_timestamp).abs()
        markers_for_calib = marker_positions.iloc[diffs.idxmin()]
    else:
        markers_for_calib = marker_positions.iloc[len(marker_positions) // 2]

    if f"{config['apriltag_marker_labels'][0]}_X" in markers_for_calib and np.isnan(
        markers_for_calib[f"{config['apriltag_marker_labels'][0]}_X"].any()
    ):
        continue

    if is_cloud_rec:
        apriltag_img = neon_rec.scene.bgr_at_time(neon_timestamp)
    else:
        apriltag_img = neon_rec.scene.data[frame].bgr

    if apriltag_img is None:
        continue

    surface_gaze_image_pts = None
    surface_gaze_object_pts = None
    if not args["surface_path"] == "":
        surface_positions = pd.read_csv(args["surface_path"])

        gaze_on_surface_x = (
            surface_positions["gaze position on surface x [normalized]"].to_numpy()
            * plane_width
        ) - plane_width / 2
        gaze_on_surface_y = (
            surface_positions["gaze position on surface y [normalized]"].to_numpy()
            * plane_height
        ) - plane_height / 2

        good_idxs = (gaze_on_surface_x < 1.0) & (gaze_on_surface_x > 0.0)
        good_gaze_on_surface_x = gaze_on_surface_x[good_idxs]
        good_gaze_on_surface_y = gaze_on_surface_y[good_idxs]

        good_gaze_on_surface_ts = surface_positions.iloc[good_idxs][
            "timestamp [ns]"
        ].to_numpy()

        surface_gaze_object_pts = np.zeros(
            (len(good_gaze_on_surface_ts), 3), dtype=np.float32
        )
        surface_gaze_object_pts[:, 0] = good_gaze_on_surface_x
        surface_gaze_object_pts[:, 1] = good_gaze_on_surface_y

        surface_gaze_object_pts = (
            config["T_neon_to_mocap"] @ surface_gaze_object_pts.T
        ).T

        good_gaze_2d = neon_rec.gaze.sample(good_gaze_on_surface_ts)

        surface_gaze_image_pts = np.zeros(
            (len(good_gaze_on_surface_ts), 2), dtype=np.float32
        )
        surface_gaze_image_pts[:, 0] = good_gaze_2d.data["point_x"]
        surface_gaze_image_pts[:, 1] = good_gaze_2d.data["point_y"]

    neon_apriltags = AprilTags(
        apriltag_detector,
        neon,
        config["apriltag_black_border_width"],
        apriltag_img,
        tag_corner_coordinates,
        config["apriltags_to_use"],
        surface_gaze_object_pts,
        surface_gaze_image_pts,
    )
    if neon_apriltags.good_detection:
        if neon_apriltags.error < smallest_error:
            smallest_error = neon_apriltags.error
            best_timestamp = neon_timestamp
            best_frame = frame
            best_plane = neon_apriltags
        else:
            continue
    else:
        continue

    neon.set_pose(neon_apriltags.pose.inverse())


if "timestamp [ns]" in marker_positions:
    diffs = (marker_positions["timestamp [ns]"] - best_timestamp).abs()
    markers_for_calib = marker_positions.iloc[diffs.idxmin()]
else:
    markers_for_calib = marker_positions.iloc[len(marker_positions) // 2]

# holds the mocap surface data for a collection of AprilTags
mocap_surface = MocapSurface()

for marker in config["apriltag_marker_labels"]:
    marker_pos_X = markers_for_calib[f"{marker}_X"].squeeze() / 1000
    marker_pos_Y = markers_for_calib[f"{marker}_Y"].squeeze() / 1000
    marker_pos_Z = markers_for_calib[f"{marker}_Z"].squeeze() / 1000

    mocap_surface.add_marker(
        MocapIRMarker(marker_pos_X, marker_pos_Y, marker_pos_Z, marker)
    )

# extract the marker positions for the head pose into a convenient object
mocap_head = MocapHead()

for id, marker in enumerate(config["neon_marker_labels"]):
    marker_pos_X = markers_for_calib[f"{marker}_X"].squeeze() / 1000
    marker_pos_Y = markers_for_calib[f"{marker}_Y"].squeeze() / 1000
    marker_pos_Z = markers_for_calib[f"{marker}_Z"].squeeze() / 1000

    mocap_head.add_marker(
        MocapIRMarker(
            marker_pos_X,
            marker_pos_Y,
            marker_pos_Z,
            id,
        )
    )

mocap_surface.construct_pose(
    config["ir_marker_radius"],
    orient_towards=np.array(
        [mocap_head.markers[0].Xs, mocap_head.markers[0].Ys, mocap_head.markers[0].Zs]
    ),
)

neon.calculate_pose_in_mocap(mocap_surface.pose)

# determine position of neon camera relative to frame markers
neon_marker_positions_in_mocap = np.array(
    [[ir_marker.Xs, ir_marker.Ys, ir_marker.Zs] for ir_marker in mocap_head.markers]
).T
avg_neon_marker_positions = np.nanmean(neon_marker_positions_in_mocap, axis=1)

print("Avg position of neon markers: ")
print(avg_neon_marker_positions)

neon_camera_position_relative_to_markers = (
    neon.pose_in_mocap.position - avg_neon_marker_positions
)

print("Neon position relative to markers: ")
print(neon_camera_position_relative_to_markers)

neon_camera_pose_relative_to_markers = Pose(
    position=neon_camera_position_relative_to_markers,
    rotation=neon.pose_in_mocap.rotation,
)

# plot tags and surface in neon camera coordinates as sanity check
plot_apriltags_in_neon(best_plane, plane_points_3d)

# plot neon's pose in display surface coordinates as sanity check
plot_neon_in_surface(neon.pose, best_plane, plane_points_3d)

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

print("\nAbsolute Neon scene camera pose in MoCap coordinates:\n")
print(neon.pose_in_mocap)

# invert ("recover") and plot neon scene camera relative to markers, as sanity check
neon_recovered = Neon(recording=neon_rec)
neon_recovered.pose_in_mocap = Pose(
    position=(
        neon_camera_pose_relative_to_markers.position + avg_neon_marker_positions
    ),
    rotation=neon.pose_in_mocap.rotation,
)
plot_neon_in_mocap(
    neon_recovered,
    mocap_surface,
    mocap_head,
    cam_z_axis_in_mocap,
)

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

# make new columns in marker_positions.csv for:
# - gaze origin in mocap coord sys at each frame
# - gaze direction in mocap coord sys at each frame

gaze_origin_Xs = np.zeros(len(marker_positions))
gaze_origin_Ys = np.zeros(len(marker_positions))
gaze_origin_Zs = np.zeros(len(marker_positions))

gaze_dir_Xs = np.zeros(len(marker_positions))
gaze_dir_Ys = np.zeros(len(marker_positions))
gaze_dir_Zs = np.zeros(len(marker_positions))

for frame in tqdm(range(len(marker_positions))):
    marker_timestamp = int(marker_positions["timestamp [ns]"].iloc[frame])

    # find the equivalent neon data based on marker timestamp
    idx = np.searchsorted(neon_rec.gaze.time, marker_timestamp)

    gaze_x = neon_rec.gaze.data["point_x"][idx]
    gaze_y = neon_rec.gaze.data["point_y"][idx]

    gaze_dir = threed_utils.unproject_points(
        np.array([gaze_x, gaze_y]),
        neon_rec.calibration.scene_camera_matrix,
        neon_rec.calibration.scene_distortion_coefficients,
        normalize=True,
    )

    gaze_dir_mocap = (gaze_dir.reshape(3, -1)).squeeze()
    gaze_dir_mocap = neon_camera_pose_relative_to_markers.rotation @ gaze_dir_mocap
    gaze_dir_Xs[frame] = gaze_dir_mocap[0]
    gaze_dir_Ys[frame] = gaze_dir_mocap[1]
    gaze_dir_Zs[frame] = gaze_dir_mocap[2]

    markers_for_calib = marker_positions.iloc[frame]

    marker_pos_X = []
    marker_pos_Y = []
    marker_pos_Z = []
    for id, marker in enumerate(config["neon_marker_labels"]):
        marker_pos_X.append(markers_for_calib[f"{marker}_X"].squeeze())
        marker_pos_Y.append(markers_for_calib[f"{marker}_Y"].squeeze())
        marker_pos_Z.append(markers_for_calib[f"{marker}_Z"].squeeze())

    avg_neon_marker_position = np.array(
        [
            np.nanmean(marker_pos_X),
            np.nanmean(marker_pos_Y),
            np.nanmean(marker_pos_Z),
        ]
    )

    gaze_origin_mocap = (
        neon_camera_position_relative_to_markers * 1000 + avg_neon_marker_position
    )
    gaze_origin_Xs[frame] = gaze_origin_mocap[0]
    gaze_origin_Ys[frame] = gaze_origin_mocap[1]
    gaze_origin_Zs[frame] = gaze_origin_mocap[2]

marker_positions["gaze_origin_X"] = gaze_origin_Xs
marker_positions["gaze_origin_Y"] = gaze_origin_Ys
marker_positions["gaze_origin_Z"] = gaze_origin_Zs

marker_positions["gaze_dir_X"] = gaze_dir_Xs
marker_positions["gaze_dir_Y"] = gaze_dir_Ys
marker_positions["gaze_dir_Z"] = gaze_dir_Zs

marker_positions.to_csv("marker_positions_w_gaze.csv")
