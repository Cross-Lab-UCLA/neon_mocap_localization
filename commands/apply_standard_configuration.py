import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import pupil_labs.neon_mocap_localization.threed_utils as threed_utils
import pupil_labs.neon_recording as plnr
from pupil_labs.neon_mocap_localization.cloud_recording import CloudRecording
from pupil_labs.neon_mocap_localization.neon import Neon

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
    "-f",
    "--neon_frame",
    "Which Neon MoCap frame was used? (EMYM => Every move you make; \
    ICTCN => I can track clearly now)",
    required=True,
)
parser.add_argument(
    "-o",
    "--output_path",
    help="The path for outputting the aligned data to CSV format.",
    default="marker_positions_w_gaze.csv",
    required=True,
)

args = vars(parser.parse_args())

# load config

config = []
with open(args["config_path"]) as f:
    config = json.load(f)

if args["neon_frame"] not in ["EMYM", "ICTCN"]:
    raise RuntimeError("The neon_frame argument must be either EMYM or ICTCN.")

# units are meters
markers_for_calib = {}
if args["neon_frame"] == "EMYM":
    markers_for_calib = {
        "Left Side": {
            "top": np.array([-104.812, 18.552, 41.677]) * 0.001,  # 5cm
            "middle": np.array([-117.39, -51.514, 30.202]) * 0.001,  # 6cm
            "bottom": np.array([-121.303, -23.053, -4.611]) * 0.001,  # 5cm
        },
        "Right Side": {
            "top": np.array([103.419, -31.267, 60.468]) * 0.001,  # 6cm
            "middle": np.array([105.787, 20.415, 27.11]) * 0.001,  # 5cm
            "bottom": np.array([130.428, -40.021, 27.11]) * 0.001,  # 6cm
        },
    }
else:  # ICTCN
    markers_for_calib = {
        "Left Side": {
            "top": np.array([-103.226, 15.0443, 45.391]) * 0.001,  # 5cm
            "middle": np.array([-136.729, -50.611, 24.09]) * 0.001,  # 6cm
            "bottom": np.array([-132.71, -9.104, 2.131]) * 0.001,  # 5cm
        },
        "Right Side": {
            "top": np.array([114.833, -48.925, 56.751]) * 0.001,  # 6cm
            "middle": np.array([128.03, -1.672, 22.702]) * 0.001,  # 5cm
            "bottom": np.array([134.237, -65.898, -1.23]) * 0.001,  # 6cm
        },
    }

centroid = np.mean(
    [
        markers_for_calib["Left Side"]["top"],
        markers_for_calib["Left Side"]["middle"],
        markers_for_calib["Left Side"]["bottom"],
        markers_for_calib["Right Side"]["top"],
        markers_for_calib["Right Side"]["middle"],
        markers_for_calib["Right Side"]["bottom"],
    ],
    axis=0,
)

v1 = markers_for_calib["Left Side"]["top"] - centroid
v1n = v1 / np.linalg.norm(v1)

v2 = markers_for_calib["Right Side"]["top"] - centroid
v2n = v2 / np.linalg.norm(v2)

v3 = np.cross(v1, v2)
v3n = v3 / np.linalg.norm(v3)

basis = np.vstack([v1n, v2n, v3n]).T

correction_vector = basis @ -centroid

theta = np.deg2rad(12)
z_axis = np.array([0, 0, 1.0])
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)],
])
cam_z_axis = basis.T @ (Rx @ z_axis)

# open neon recording and initialize neon object

neon_rec: CloudRecording | plnr.NeonRecording
is_cloud_rec = False
try:
    neon_rec = CloudRecording(args["neon_rec_path"])
    is_cloud_rec = True
    nframes = neon_rec.scene.nframes
except Exception:
    try:
        neon_rec = plnr.open(args["neon_rec_path"])
        nframes = len(neon_rec.scene.data)
    except Exception as e:
        raise ValueError("Not a valid Neon data directory") from e

neon = Neon(recording=neon_rec)

# load mocap data

marker_positions = pd.read_csv(args["mocap_path"])

# make new columns in marker_positions.csv for:
# - gaze origin in mocap coord sys at each frame
# - gaze direction in mocap coord sys at each frame

gaze_origins = np.zeros((len(marker_positions), 3))
gaze_dirs = np.zeros((len(marker_positions), 3))

for frame in tqdm(range(len(marker_positions))):
    marker_timestamp = int(marker_positions["timestamp [ns]"].iloc[frame])

    # find the equivalent neon data based on marker timestamp
    idx = np.searchsorted(neon_rec.gaze.time, marker_timestamp)

    gaze_x = neon_rec.gaze.data["point_x"][idx]
    gaze_y = neon_rec.gaze.data["point_y"][idx]

    if neon_rec.calibration is not None:
        gaze_dir = threed_utils.unproject_points(
            np.array([gaze_x, gaze_y]),
            neon_rec.calibration.scene_camera_matrix,
            neon_rec.calibration.scene_distortion_coefficients,
            normalize=True,
        )
    else:
        raise ValueError("Recording has no camera calibration data.")

    gaze_dir = (gaze_dir.reshape(3, -1)).squeeze()

    markers_for_calib = marker_positions.iloc[frame]

    curr_neon_markers = []
    for marker_name in [
        "Left Top",
        "Left Middle",
        "Left Bottom",
        "Right Top",
        "Right Middle",
        "Right Bottom",
    ]:
        marker = config["neon_marker_labels"][marker_name]
        curr_neon_markers.append(
            np.array([
                markers_for_calib[f"{marker}_X"].squeeze()
                * config["mocap_unit_conversion_factor"],
                markers_for_calib[f"{marker}_Y"].squeeze()
                * config["mocap_unit_conversion_factor"],
                markers_for_calib[f"{marker}_Z"].squeeze()
                * config["mocap_unit_conversion_factor"],
            ])
        )

    centroid = np.mean(np.array(curr_neon_markers), axis=0)

    v1 = curr_neon_markers[0] - centroid
    v1n = v1 / np.linalg.norm(v1)

    v2 = curr_neon_markers[3] - centroid
    v2n = v2 / np.linalg.norm(v2)

    v3 = np.cross(v1, v2)
    v3n = v3 / np.linalg.norm(v3)

    basis = np.vstack([v1n, v2n, v3n]).T

    gaze_origins[frame, :] = (centroid + basis.T @ correction_vector) / config[
        "mocap_unit_conversion_factor"
    ]

    elevation, azimuth = threed_utils.cartesian_to_spherical(gaze_dir)
    elevation = np.deg2rad(elevation)
    azimuth = np.deg2rad(azimuth)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(elevation), -np.sin(elevation)],
        [0, np.sin(elevation), np.cos(elevation)],
    ])

    Rz = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0],
        [np.sin(azimuth), np.cos(azimuth), 0],
        [0, 0, 1],
    ])

    gaze_dir_mocap = Rz @ Rx @ (basis.T @ cam_z_axis)
    gaze_dirs[frame, :] = gaze_dir_mocap


marker_positions["gaze_origin_X"] = gaze_origins[:, 0]
marker_positions["gaze_origin_Y"] = gaze_origins[:, 1]
marker_positions["gaze_origin_Z"] = gaze_origins[:, 2]

marker_positions["gaze_dir_X"] = gaze_dirs[:, 0]
marker_positions["gaze_dir_Y"] = gaze_dirs[:, 1]
marker_positions["gaze_dir_Z"] = gaze_dirs[:, 2]

# export data

marker_positions.to_csv(args["output_path"])
