import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyxdf  #  type: ignore
import scipy.io as sio
from ezc3d import c3d  #  type: ignore
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import interp1d

import pupil_labs.neon_recording as plnr
from pupil_labs.neon_mocap_localization.threed_utils import unproject_points


def normalize(x: float) -> np.floating:
    return (x - np.mean(x)) / np.std(x)


def align_signals(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    y_ts: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], list[int], np.signedinteger]:
    if np.isnan(x).any():
        x[np.isnan(x)] = np.nanmean(x)

    if np.isnan(y).any():
        y[np.isnan(y)] = np.nanmean(y)

    # x_norm = normalize(x)
    # y_norm = normalize(y)

    n = len(x)

    windows = sliding_window_view(y, n)
    sq_diff = np.sum((windows - x) ** 2, axis=1)

    # max_corr_idx = np.argmax(np.correlate(y, x, mode="valid"))
    max_corr_idx = np.argmin(sq_diff)

    x_time_in_y = y_ts[max_corr_idx : (max_corr_idx + len(x))]
    x_idxs_in_y = list(range(max_corr_idx, (max_corr_idx + len(x))))

    return x_time_in_y, x_idxs_in_y, max_corr_idx


# parse args

parser = argparse.ArgumentParser(
    description="Determines relative position of Neon scene camera in MoCap \
coordinate system"
)

parser.add_argument(
    "-m",
    "--mocap_mat_path",
    help="The path to the Qualisys data (MAT file)",
    required=True,
)
parser.add_argument(
    "-c3", "--c3d_path", help="The path to the Qualisys data (C3D file)", required=True
)
parser.add_argument(
    "-r",
    "--neon_rec_path",
    help="The path to the associated Neon recording",
    required=True,
)
parser.add_argument(
    "-x",
    "--xdf_path",
    help="The XDF file produced by LabRecorder of Lab Streaming Layer",
)
parser.add_argument(
    "-o",
    "--output_path",
    help="The CSV file path for saving the converted data (in Neon timebase)",
    default="marker_positions.csv",
)
parser.add_argument(
    "-c",
    "--config_path",
    help="A config file containing the parameters that remain constant between \
sessions.",
    required=True,
)
parser.add_argument(
    "-tb",
    "--trim_begin",
    help="Trim this many datapoints from the beginning of the MoCap data. Can \
help with syncing streams.",
)
parser.add_argument(
    "-te",
    "--trim_end",
    help="Trim this many datapoints from the end of the MoCap data. Can help \
with syncing streams.",
)

args = vars(parser.parse_args())

mocap_mat_path = args["mocap_mat_path"]
xdf_path = args["xdf_path"]
neon_rec_path = args["neon_rec_path"]
c3d_path = args["c3d_path"]
output_path = args["output_path"]

config = []
with open(args["config_path"]) as f:
    config = json.load(f)

# load qualisys data

data = sio.loadmat(mocap_mat_path)
xdf_data = pyxdf.load_xdf(xdf_path)
c3d_data = c3d(c3d_path)

# load neon data

neon_rec = plnr.open(neon_rec_path)

# extract relevant marker positions from mocap data

condition_name = list(data.keys())[-1]
try:
    marker_positions = data[condition_name][0][0][5][0][0][0][0][0][2]
except Exception:
    marker_positions = data[condition_name][0][0][6][0][0][0][0][0][2]

nsamples = marker_positions.shape[2]

# Indices of relevant markers in marker_positions array

try:
    marker_names = data[condition_name][0][0][5][0][0][0][0][0][1][0]
except Exception:
    marker_names = data[condition_name][0][0][6][0][0][0][0][0][1][0]

marker_indices = {str(name[0]): idx for idx, name in enumerate(marker_names)}

# timesync with neon data

reference_positions = marker_positions[
    marker_indices[config["qualisys_reference_marker"]]
].squeeze()

trim_begin = int(args["trim_begin"]) if args["trim_begin"] else 0
trim_end = -int(args["trim_end"]) if args["trim_end"] else len(reference_positions)

if reference_positions.shape[1] == 4:
    reference_positions = reference_positions.T

reference_duration = reference_positions.shape[-1] / 200
reference_timestamps = np.arange(0, reference_duration, 1 / 200)

qualisys_xdf_idx = -1
neon_xdf_idx = -1
for idx, channel in enumerate(xdf_data[0]):
    if channel["info"]["name"][0] == "Qualisys":
        qualisys_xdf_idx = idx
    elif channel["info"]["name"][0] == "Neon Companion_Neon Gaze":
        neon_xdf_idx = idx


reference_xdf_data_idx = -1
for idx, channel in enumerate(
    xdf_data[0][qualisys_xdf_idx]["info"]["desc"][0]["channels"][0]["channel"]
):
    if config["qualisys_reference_marker"] in channel["label"][0]:
        reference_xdf_data_idx = idx
        break


reference_timestamps_xdf = xdf_data[0][qualisys_xdf_idx]["time_stamps"]
reference_positions_xdf = (
    xdf_data[0][qualisys_xdf_idx]["time_series"][
        :, reference_xdf_data_idx : reference_xdf_data_idx + 3
    ].squeeze()
    * 1000
)  # convert to millimeters

# determine which part of QTM data the LSL recording corresponds
# to via cross-correlation alignment

time_qtm_in_xdf, idxs_qtm_in_xdf, qtm_offset = align_signals(
    reference_positions[0, trim_begin:trim_end].squeeze(),
    reference_positions_xdf[:, 0].squeeze(),
    reference_timestamps_xdf,
)

qtm_start_in_xdf = qtm_offset - trim_begin
qtm_end_in_xdf = qtm_start_in_xdf + len(reference_positions[0, :].squeeze())
full_time_qtm_in_xdf = reference_timestamps_xdf[qtm_start_in_xdf:qtm_end_in_xdf]

plt.plot(reference_timestamps_xdf, reference_positions_xdf[:, 0].squeeze())
plt.plot(
    full_time_qtm_in_xdf,
    # reference_positions[
    #     0, len(reference_positions[0, :]) - len(time_qtm_in_xdf) :
    # ].squeeze(),
    reference_positions[0, : len(full_time_qtm_in_xdf)].squeeze(),
)
plt.show()

# determine offset between neon and LSL recording via
# cross-correlation alignment

time_gaze_in_xdf, idxs_gaze_in_xdf, neon_offset = align_signals(
    neon_rec.gaze.data["point_x"],
    xdf_data[0][neon_xdf_idx]["time_series"][:, 0],
    xdf_data[0][neon_xdf_idx]["time_stamps"],
)

plt.plot(
    xdf_data[0][neon_xdf_idx]["time_stamps"],
    xdf_data[0][neon_xdf_idx]["time_series"][:, 0],
)
plt.plot(
    time_gaze_in_xdf[: len(neon_rec.gaze.data["point_x"])],
    neon_rec.gaze.data["point_x"],
)
plt.show()

time_gaze_in_xdf = time_gaze_in_xdf[: len(neon_rec.gaze.data["point_x"])]

plt.plot(
    full_time_qtm_in_xdf,
    # reference_positions[
    #     0, len(reference_positions[0, :]) - len(time_qtm_in_xdf) :
    # ].squeeze(),
    reference_positions[0, : len(full_time_qtm_in_xdf)].squeeze(),
)
plt.plot(time_gaze_in_xdf, neon_rec.gaze.data["point_x"])
plt.show()

# marker_positions = marker_positions[
#     :, :, len(reference_positions[0, :]) - len(time_qtm_in_xdf) :
# ]
marker_positions = marker_positions[:, :, : len(full_time_qtm_in_xdf)]

# make dataframe to export as csv

marker_df = pd.DataFrame()
marker_df["timestamp [ns]"] = neon_rec.gaze.time
# for marker in config["apriltag_marker_labels"] + config["neon_marker_labels"]:
for marker_name in marker_names:
    marker = str(marker_name[0])
    index = marker_indices[marker]

    marker_pos = marker_positions[index].squeeze()
    # marker_pos = marker_pos[:, trim_begin:trim_end]

    if marker_pos.shape[1] == 4:
        marker_pos = marker_pos.T

    # re-interpolate qtm data to correspond exactly to neon data

    fx = interp1d(
        full_time_qtm_in_xdf,
        marker_pos[0, :],
        bounds_error=False,
        fill_value=np.nan,
    )
    fy = interp1d(
        full_time_qtm_in_xdf,
        marker_pos[1, :],
        bounds_error=False,
        fill_value=np.nan,
    )
    fz = interp1d(
        full_time_qtm_in_xdf,
        marker_pos[2, :],
        bounds_error=False,
        fill_value=np.nan,
    )

    marker_pos_x_in_neon = fx(time_gaze_in_xdf)
    marker_pos_y_in_neon = fy(time_gaze_in_xdf)
    marker_pos_z_in_neon = fz(time_gaze_in_xdf)

    marker_df[f"{marker}_X"] = list(marker_pos_x_in_neon)
    marker_df[f"{marker}_Y"] = list(marker_pos_y_in_neon)
    marker_df[f"{marker}_Z"] = list(marker_pos_z_in_neon)

# export csv

marker_df.to_csv(output_path)

# re-interpolate neon data to correspond exactly to qtm timestamps

if neon_rec.calibration is not None:
    gaze_3d = unproject_points(
        neon_rec.gaze.data[["point_x", "point_y"]],
        neon_rec.calibration.scene_camera_matrix,
        neon_rec.calibration.scene_distortion_coefficients,
        normalize=True,
    )
else:
    raise ValueError("Recording has no camera calibration data.")

f_gaze_x = interp1d(
    time_gaze_in_xdf,
    gaze_3d[:, 0].squeeze(),
    bounds_error=False,
    fill_value=np.nan,
)
f_gaze_y = interp1d(
    time_gaze_in_xdf,
    gaze_3d[:, 1].squeeze(),
    bounds_error=False,
    fill_value=np.nan,
)
f_gaze_z = interp1d(
    time_gaze_in_xdf,
    gaze_3d[:, 2].squeeze(),
    bounds_error=False,
    fill_value=np.nan,
)

gaze_x_in_qtm = f_gaze_x(time_qtm_in_xdf)
gaze_y_in_qtm = f_gaze_y(time_qtm_in_xdf)
gaze_z_in_qtm = f_gaze_z(time_qtm_in_xdf)

gaze_xyz_in_qtm = np.column_stack([gaze_x_in_qtm, gaze_y_in_qtm, gaze_z_in_qtm])

# export C3D file with synced neon data

# make a new c3d_data file based on the original

c3d_points = c3d_data["data"]["points"]
nframes_qtm = c3d_points.shape[2]

gaze_for_c3d = np.zeros(
    (4, 1, nframes_qtm),
    dtype=float,
)

# homogeneous coordinates
gaze_for_c3d[0, 0, len(reference_positions[0, :]) - len(time_qtm_in_xdf) :] = (
    gaze_xyz_in_qtm[:, 0]
)  # X
gaze_for_c3d[1, 0, len(reference_positions[0, :]) - len(time_qtm_in_xdf) :] = (
    gaze_xyz_in_qtm[:, 1]
)  # Y
gaze_for_c3d[2, 0, len(reference_positions[0, :]) - len(time_qtm_in_xdf) :] = (
    gaze_xyz_in_qtm[:, 2]
)  # Z
gaze_for_c3d[3, 0, len(reference_positions[0, :]) - len(time_qtm_in_xdf) :] = 1.0  # W

c3d_points_new = np.concatenate([c3d_points, gaze_for_c3d], axis=1)
c3d_data["data"]["points"] = c3d_points_new

params = c3d_data["parameters"]

labels = list(params["POINT"]["LABELS"]["value"])
labels.append("Gaze-3D-Vector")
params["POINT"]["LABELS"]["value"] = labels

if "LONG_NAMES" in params["POINT"]:
    long_names = list(params["POINT"]["LONG_NAMES"]["value"])
    long_names.append("Gaze-3D-Vector")
    params["POINT"]["LONG_NAMES"]["value"] = long_names

params["POINT"]["USED"]["value"] = [c3d_points_new.shape[1]]

# make dummy residuals and camera_masks for the gaze data
c3d_meta = c3d_data["data"]["meta_points"]
c3d_meta_new = {}
c3d_meta_new["residuals"] = np.concatenate(
    [c3d_meta["residuals"], np.ones((1, 1, nframes_qtm))], axis=1
)
c3d_meta_new["camera_masks"] = np.concatenate(
    [c3d_meta["camera_masks"], np.zeros((7, 1, nframes_qtm), dtype=np.bool)], axis=1
)
c3d_data["data"]["meta_points"] = c3d_meta_new

c3d_data.write("mocap_with_gaze.c3d")
print("Wrote mocap_with_gaze.c3d")
