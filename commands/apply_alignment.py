import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import pupil_labs.neon_mocap_localization.threed_utils as threed_utils
import pupil_labs.neon_recording as plnr
from pupil_labs.neon_mocap_localization.cloud_recording import CloudRecording
from pupil_labs.neon_mocap_localization.mocap import MocapIRMarker

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

# load mocap data

marker_positions = pd.read_csv(args["mocap_path"])
neutral_mocap_heading = np.array(config["neutral_heading"])

# load calibration data

with open(args["calibration_name"], "rb") as file:
    calibration_data = pickle.load(file)  # noqa: S301

    neon_camera_pose_relative_to_markers = calibration_data[
        "neon_camera_pose_relative_to_markers"
    ]
    neon = calibration_data["neon"]
    mocap_head = calibration_data["mocap_head"]

# make new columns in marker_positions.csv for:
# - gaze origin in mocap coord sys at each frame
# - gaze direction in mocap coord sys at each frame

gaze_origins = np.zeros((len(marker_positions), 3))
gaze_dirs = np.zeros((len(marker_positions), 3))
head_dirs = np.zeros((len(marker_positions), 3))

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

    gaze_dir_mocap = (gaze_dir.reshape(3, -1)).squeeze()
    gaze_dir_mocap = neon_camera_pose_relative_to_markers.rotation @ gaze_dir_mocap

    markers_for_calib = marker_positions.iloc[frame]

    curr_neon_markers = []
    for marker in config["neon_marker_labels"]:
        curr_neon_markers.append(  # noqa: PERF401
            MocapIRMarker(
                markers_for_calib[f"{marker}_X"].squeeze()
                * config["mocap_unit_conversion_factor"],
                markers_for_calib[f"{marker}_Y"].squeeze()
                * config["mocap_unit_conversion_factor"],
                markers_for_calib[f"{marker}_Z"].squeeze()
                * config["mocap_unit_conversion_factor"],
                marker,
            )
        )

    if not neon.update_neon_camera_pose(curr_neon_markers, mocap_head):
        gaze_origins[frame, :] = np.nan
        gaze_dirs[frame, :] = np.nan
    else:
        if config["flip_gaze_x"]:
            gaze_dir_mocap[0] *= -1

        if config["flip_gaze_y"]:
            gaze_dir_mocap[1] *= -1

        if config["flip_gaze_z"]:
            gaze_dir_mocap[2] *= -1

        gaze_origins[frame, :] = (
            neon.transformed_pose_in_mocap.position
            / config["mocap_unit_conversion_factor"]
        )

        gaze_dir_mocap = neon.transformed_pose_in_mocap.rotation @ gaze_dir_mocap
        gaze_dirs[frame, :] = gaze_dir_mocap

        head_dirs[frame, :] = (
            neon.transformed_pose_in_mocap.rotation @ neutral_mocap_heading
        )

marker_positions["gaze_origin_X"] = gaze_origins[:, 0]
marker_positions["gaze_origin_Y"] = gaze_origins[:, 1]
marker_positions["gaze_origin_Z"] = gaze_origins[:, 2]

marker_positions["gaze_dir_X"] = gaze_dirs[:, 0]
marker_positions["gaze_dir_Y"] = gaze_dirs[:, 1]
marker_positions["gaze_dir_Z"] = gaze_dirs[:, 2]

marker_positions["head_dir_X"] = head_dirs[:, 0]
marker_positions["head_dir_Y"] = head_dirs[:, 1]
marker_positions["head_dir_Z"] = head_dirs[:, 2]

marker_positions.to_csv(
    str(Path(args["mocap_path"]).parent)
    + "/"
    + Path(args["mocap_path"]).stem
    + "_with_gaze_neon_ts.csv"
)
