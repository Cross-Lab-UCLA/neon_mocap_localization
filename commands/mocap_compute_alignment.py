import argparse
import json
import pickle

import numpy as np
import pandas as pd
from pupil_apriltags import Detector  # type: ignore
from tqdm import tqdm

import pupil_labs.neon_recording as plnr
from pupil_labs.neon_mocap_localization.apriltags import AprilTags
from pupil_labs.neon_mocap_localization.cloud_recording import CloudRecording
from pupil_labs.neon_mocap_localization.mocap import (
    MocapHead,
    MocapIRMarker,
    MocapSurface,
)
from pupil_labs.neon_mocap_localization.neon import Neon
from pupil_labs.neon_mocap_localization.plots import (
    plot_apriltags_in_neon,
    plot_neon_in_mocap,
    plot_neon_in_surface,
    plot_surface_local_coordinate_system_in_mocap,
)
from pupil_labs.neon_mocap_localization.pose import Pose

parser = argparse.ArgumentParser(
    description="Determines trajectory of Neon scene camera in MoCap coordinate system"
)

parser.add_argument(
    "-r",
    "--neon_rec_path",
    help="The path to the Neon Recording Data (Native and Timeseries accepted)",
    required=True,
)
parser.add_argument(
    "-m",
    "--mocap_path",
    help="The path to the MoCap data (CSV; in Neon timebase (i.e., synced))",
    required=True,
)
parser.add_argument(
    "-c",
    "--config_path",
    help="A config file containing the parameters that remain constant between \
sessions.",
    required=True,
)
parser.add_argument(
    "-x",
    "--calibration_name",
    help="The base name of the file with the calibration data (i.e., without \
the '.pkl' extension)",
    default="calibration_data.pkl",
)

args = vars(parser.parse_args())

# load config

config = []
with open(args["config_path"]) as f:
    config = json.load(f)

# load data

# open neon recording and initialize neon object

neon_rec: CloudRecording | plnr.NeonRecording
try:
    neon_rec = CloudRecording(args["neon_rec_path"])
    nframes = neon_rec.scene.nframes
except Exception:
    try:
        neon_rec = plnr.open(args["neon_rec_path"])
        nframes = len(neon_rec.scene.data)
    except Exception as e:
        raise ValueError("Not a valid Neon data directory") from e

# load mocap data

marker_positions = pd.read_csv(args["mocap_path"])

# load apriltag corner data (in a user-specified local coordinate system)

plane_width = config["apriltag_pattern_width"]
plane_height = config["apriltag_pattern_height"]

# tags should be listed in order of increasing ID
# should be tag 0, tag 1, tag 2, tag 3
# order for corners of each tag should be -> BL BR TR TL
# x, y only
# units should be meters
tag_corner_coordinates = config["apriltag_corner_local_coordinates"]
if tag_corner_coordinates:
    for k, v in tag_corner_coordinates.items():
        m = np.zeros((len(v), 3), dtype=np.float32)
        v = np.array(v) * config["corner_unit_conversion_factor"]

        m[:, :2] = v

        tag_corner_coordinates[k] = m

    for _k, v in tag_corner_coordinates.items():
        v[:, 0] -= plane_width / 2
        v[:, 1] -= plane_height / 2

        # for c in range(len(tag_corner_coordinates[k])):
        # tag_corner_coordinates[k][c] = np.array(config["T_neon_to_mocap"]) @ v[c]


plane_points_3d = np.array([
    [-plane_width / 2, plane_height / 2, 0],  # BL
    [plane_width / 2, plane_height / 2, 0],  # BR
    [plane_width / 2, -plane_height / 2, 0],  # TR
    [-plane_width / 2, -plane_height / 2, 0],  # TL
    [-plane_width / 2, plane_height / 2, 0],  # BL
])
# for c in range(len(plane_points_3d)):
# plane_points_3d[c] = np.array(config["T_neon_to_mocap"]) @ plane_points_3d[c]

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
smallest_error = np.inf
best_timestamp = 0
best_frame = 0
best_plane: AprilTags
for frame in tqdm(range(int(nframes))):
    neon_timestamp = neon_rec.scene.time[frame]

    if (
        neon_timestamp < marker_positions["timestamp [ns]"].iloc[0]
        or neon_timestamp > marker_positions["timestamp [ns]"].iloc[-1]
    ):
        continue

    # find the equivalent marker positions based on neon timestamp
    diffs = (marker_positions["timestamp [ns]"] - neon_timestamp).abs()  # type: ignore
    markers_for_calib = marker_positions.iloc[diffs.idxmin()]  # type: ignore

    good_marker_data = True
    if (
        f"{config['apriltag_marker_labels']['Top Left']}_X" in markers_for_calib.keys()  # noqa: SIM118
    ):
        for marker_corners in ["Top Left", "Top Right", "Bottom Right", "Bottom Left"]:
            for coord in ["X", "Y", "Z"]:
                if np.isnan(
                    markers_for_calib[
                        f"{config['apriltag_marker_labels'][marker_corners]}_{coord}"
                    ]
                ):
                    good_marker_data = False

    if (
        f"{config['neon_marker_labels'][0]}_X" in markers_for_calib.keys()  # noqa: SIM118
    ):
        for neon_marker in config["neon_marker_labels"]:
            for coord in ["X", "Y", "Z"]:
                if np.isnan(markers_for_calib[f"{neon_marker}_{coord}"]):
                    good_marker_data = False

    if not good_marker_data:
        continue

    if isinstance(neon_rec, CloudRecording):
        apriltag_img = neon_rec.scene.bgr_at_time(neon_timestamp)
    else:
        apriltag_img = neon_rec.scene.data[frame].bgr

    if apriltag_img is None:
        continue

    neon_apriltags = AprilTags(
        apriltag_detector,
        neon,
        config["apriltag_black_side_length"],
        apriltag_img,
        tag_corner_coordinates,
        config["apriltags_to_use"],
    )
    if neon_apriltags.good_detection:
        if neon_apriltags.error < smallest_error:
            smallest_error = neon_apriltags.error
            best_timestamp = neon_timestamp
            best_frame = frame
            best_plane = neon_apriltags

            neon.set_pose(neon_apriltags.pose.inverse())
        else:
            continue
    else:
        continue


diffs = (marker_positions["timestamp [ns]"] - best_timestamp).abs()  # type: ignore
markers_for_calib = marker_positions.iloc[diffs.idxmin()]  # type: ignore

# holds the mocap surface data for a collection of AprilTags
mocap_surface = MocapSurface()

for marker_id in ["Top Left", "Top Right", "Bottom Right", "Bottom Left"]:
    marker = config["apriltag_marker_labels"][marker_id]

    marker_pos_X = (
        markers_for_calib[f"{marker}_X"].squeeze()
        * config["mocap_unit_conversion_factor"]
    )
    marker_pos_Y = (
        markers_for_calib[f"{marker}_Y"].squeeze()
        * config["mocap_unit_conversion_factor"]
    )
    marker_pos_Z = (
        markers_for_calib[f"{marker}_Z"].squeeze()
        * config["mocap_unit_conversion_factor"]
    )

    mocap_surface.add_marker(
        MocapIRMarker(marker_pos_X, marker_pos_Y, marker_pos_Z, marker_id)
    )

# extract the marker positions for the head pose into a convenient object
mocap_head = MocapHead()

for marker in config["neon_marker_labels"]:
    marker_pos_X = (
        markers_for_calib[f"{marker}_X"].squeeze()
        * config["mocap_unit_conversion_factor"]
    )
    marker_pos_Y = (
        markers_for_calib[f"{marker}_Y"].squeeze()
        * config["mocap_unit_conversion_factor"]
    )
    marker_pos_Z = (
        markers_for_calib[f"{marker}_Z"].squeeze()
        * config["mocap_unit_conversion_factor"]
    )

    mocap_head.add_marker(
        MocapIRMarker(
            marker_pos_X,
            marker_pos_Y,
            marker_pos_Z,
            marker,
        )
    )

mocap_surface.construct_pose_simple(
    config["ir_marker_radius"],
    np.array(config["T_mocap_to_apriltag"]),
    orient_towards=np.array([
        mocap_head.markers[0].Xs,
        mocap_head.markers[0].Ys,
        mocap_head.markers[0].Zs,
    ]),
)

mocap_head.get_local_coord_sys()

neon.calculate_reference_pose_in_mocap(mocap_surface.pose)

# determine position of neon camera relative to frame markers
neon_marker_positions_in_mocap = np.array([
    [ir_marker.Xs, ir_marker.Ys, ir_marker.Zs] for ir_marker in mocap_head.markers
]).T

avg_neon_marker_position = np.nanmean(neon_marker_positions_in_mocap, axis=1)

neon_camera_position_relative_to_markers = (
    neon.reference_pose_in_mocap.position - avg_neon_marker_position
)

print("Neon position relative to markers: ")
print(neon_camera_position_relative_to_markers)

neon_camera_pose_relative_to_markers = Pose(
    position=neon_camera_position_relative_to_markers,
    rotation=neon.reference_pose_in_mocap.rotation,
)

# plot tags and surface in neon camera coordinates as sanity check
plot_apriltags_in_neon(best_plane, plane_points_3d)

# plot neon's pose in display surface coordinates as sanity check
plot_neon_in_surface(neon.pose, best_plane, plane_points_3d)

# plot surface local coordinate system in mocap space,
# as obtained via SVD, as sanity check
plot_surface_local_coordinate_system_in_mocap(mocap_surface)

# plot the final positions, as sanity check
plot_neon_in_mocap(
    neon,
    mocap_surface,
    mocap_head,
)

print("\nAbsolute Neon scene camera pose in MoCap coordinates:\n")
print(neon.reference_pose_in_mocap)

neon.update_neon_camera_pose(mocap_head.markers, mocap_head)

plot_neon_in_mocap(
    neon,
    mocap_surface,
    mocap_head,
)

# Export calibration data
with open(args["calibration_name"], "wb") as file:
    data = {
        "neon_camera_pose_relative_to_markers": neon_camera_pose_relative_to_markers,
        "mocap_head": mocap_head,
        "neon": neon,
    }
    pickle.dump(data, file)
    print(f"Data has been pickled and saved to {args['calibration_name']}")
