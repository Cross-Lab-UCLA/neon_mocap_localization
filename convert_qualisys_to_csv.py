import argparse

import numpy as np
import pandas as pd
import scipy.io as sio
import pyxdf
import matplotlib.pyplot as plt
import pupil_labs.neon_recording as plnr


def masked_corr_at_shift(s, x_short, x_long, mask_short, mask_long, valid_short_idx):
    """
    Correlation between x_short and x_long at shift s, using only indices
    where BOTH signals are non-NaN.
    """
    N = len(x_long)

    idx_long = valid_short_idx + s
    in_bounds = idx_long < N
    idx_long = idx_long[in_bounds]
    idx_short = valid_short_idx[in_bounds]

    both_valid = mask_long[idx_long]
    idx_long = idx_long[both_valid]
    idx_short = idx_short[both_valid]

    n = len(idx_long)
    if n < 500:  # require enough overlapping samples
        return np.nan

    a = x_long[idx_long].astype(float)
    b = x_short[idx_short].astype(float)

    a_mean, b_mean = np.nanmean(a), np.nanmean(b)
    a_std, b_std = np.nanstd(a), np.nanstd(b)
    if a_std == 0 or b_std == 0:
        return np.nan

    a = (a - a_mean) / a_std
    b = (b - b_mean) / b_std

    return float(np.nanmean(a * b))


def align_and_build_timestamps(t_short, x_short, x_long):
    """
    Find alignment between short (timestamped) and long (untimestamped) signals,
    then build timestamps for the long signal.

    Returns
    -------
    start_idx : int
        Index in x_long where x_short[0] aligns.
    a, b : float
        Time mapping parameters: t(k) ≈ a + b * k
    t_long_est : np.ndarray
        Estimated timestamps for every sample in x_long.
    """
    N = len(x_long)
    M = len(x_short)

    mask_short = ~np.isnan(x_short)
    mask_long = ~np.isnan(x_long)
    valid_short_idx = np.where(mask_short)[0]

    # Scan all shifts where short fits in long
    num_shifts = N - M + 1
    corr_vals = np.empty(num_shifts, dtype=float)
    for s in range(num_shifts):
        corr_vals[s] = masked_corr_at_shift(
            s, x_short, x_long, mask_short, mask_long, valid_short_idx
        )

    start_idx = int(np.nanargmax(corr_vals))

    # Fit t_short ≈ a + b * k_long over the overlap
    dt = 1.0 / 200.0  # assume a perfect 200 Hz for Qualisys data

    k_overlap = start_idx + np.arange(M)
    a = np.mean(t_short - dt * k_overlap)

    k_all = np.arange(N)
    t_long_est = a + dt * k_all

    return start_idx, a, t_long_est


def plot_alignment(t_long_est, x_long, t_short, x_short, start_idx):
    N = len(x_long)
    M = len(x_short)
    mask_short = ~np.isnan(x_short)
    mask_long = ~np.isnan(x_long)

    # Indices of the overlap in the long signal
    overlap_long_idx = np.arange(start_idx, start_idx + M)
    overlap_long_idx = overlap_long_idx[overlap_long_idx < N]

    # Only plot points where both are valid
    overlap_short_idx = overlap_long_idx - start_idx
    both_valid = mask_long[overlap_long_idx] & mask_short[overlap_short_idx]

    idx_long_plot = overlap_long_idx[both_valid]
    idx_short_plot = overlap_short_idx[both_valid]

    plt.plot(t_long_est[idx_long_plot], x_long[idx_long_plot], "k-")
    plt.plot(
        t_short[idx_short_plot],
        x_short[idx_short_plot],
        "r-",
    )
    plt.tight_layout()
    plt.show()


# parse args

parser = argparse.ArgumentParser(
    description="Determines relative position of Neon scene camera in MoCap coordinate system"
)
parser.add_argument(
    "-m",
    "--mocap_path",
    help="The path to the Qualisys data",
    required=True,
)
parser.add_argument(
    "-r",
    "--neon_rec_path",
    help="The path to the associated Neon recording",
    required=True,
)
parser.add_argument(
    "-x", "--xdf_path", help="The XDF file produced by Lab Streaming Layer"
)
parser.add_argument(
    "-nm", "--neon_markers", help="The number of markers on the wearers head", default=6
)
parser.add_argument(
    "-o",
    "--output_path",
    help="The CSV file path for saving the converted data (in Neon timebase)",
    default="marker_positions.csv",
)

args = vars(parser.parse_args())

mocap_path = args["mocap_path"]
xdf_path = args["xdf_path"]
neon_rec_path = args["neon_rec_path"]
num_neon_markers = args["neon_markers"]
output_path = args["output_path"]

# load qualisys data

data = sio.loadmat(mocap_path)
xdf_data = pyxdf.load_xdf(xdf_path)
neon_rec = plnr.open(neon_rec_path)

# extract relevant marker positions from mocap data

marker_positions = data["Static_Test"][0][0][5][0][0][0][0][0][2]
nsamples = marker_positions.shape[2]

# Indices of relevant markers in marker_positions array

marker_indices = {
    "RTRAG": 0,
    "LTRAG": 2,
    "RMAST": 1,
    "LMAST": 3,
    "T1TL": 31 - 1,
    "T1TR": 41 - 1,
    "T1BR": 37 - 1,
    "T1BL": 32 - 1,
    "T2TL": 39 - 1,
    "T2TR": 40 - 1,
    "T2BL": 29 - 1,
    "T2BR": 33 - 1,
    "T3TL": 35 - 1,
    "T3TR": 38 - 1,
    "T3BL": 34 - 1,
    "T3BR": 36 - 1,
}

# timesync with neon data

rtrag_positions = marker_positions[marker_indices["RTRAG"]].squeeze()

# xdf_data[0][0]["info"]["desc"][0]["channels"][0]["channel"][0]
# xdf_data[0][0]["info"]["desc"][0]["channels"][0]["channel"][1]
# xdf_data[0][0]["info"]["desc"][0]["channels"][0]["channel"][2]
rtrag_timestamps_xdf = xdf_data[0][0]["time_stamps"]
rtrag_positions_xdf = (
    xdf_data[0][0]["time_series"][:, :3].squeeze() * 1000
)  # convert to millimeters

# determine which part of QTM data the LSL recording corresponds
# to via cross-correlation alignment

mocap_idx_offset, mocap_time_offset, time_qtm_in_xdf = align_and_build_timestamps(
    rtrag_timestamps_xdf,
    rtrag_positions_xdf[:, 0].squeeze(),
    rtrag_positions[0, :].squeeze(),
)

plot_alignment(
    time_qtm_in_xdf,
    rtrag_positions[0, :].squeeze(),
    rtrag_timestamps_xdf,
    rtrag_positions_xdf[:, 0].squeeze(),
    mocap_idx_offset,
)

# determine offset between neon and LSL recording via
# cross-correlation alignment

gaze_idx_offset, gaze_time_offset, time_gaze_in_xdf = align_and_build_timestamps(
    xdf_data[0][1]["time_stamps"],
    xdf_data[0][1]["time_series"][:, 0],
    neon_rec.gaze.data["point_x"],
)

plot_alignment(
    time_gaze_in_xdf,
    neon_rec.gaze.data["point_x"],
    xdf_data[0][1]["time_stamps"],
    xdf_data[0][1]["time_series"][:, 0],
    gaze_idx_offset,
)

plt.plot(time_qtm_in_xdf, rtrag_positions[0, :].squeeze())
plt.plot(time_gaze_in_xdf, neon_rec.gaze.data["point_x"])
plt.show()

good_rtrag_indices = ~np.isnan(rtrag_positions[0, :])
qtm_indices_in_gaze = np.searchsorted(
    time_gaze_in_xdf, time_qtm_in_xdf[good_rtrag_indices]
)
qtm_indices_in_scene = np.unique(
    np.searchsorted(neon_rec.scene.time, neon_rec.gaze.time[qtm_indices_in_gaze])
)

time_qtm_in_neon = neon_rec.gaze.time[qtm_indices_in_gaze]

# make dataframe to export as csv

cols = [
    f"T{tc}{corner}"
    for tc in ["1", "2", "3"]  # , "4"]
    for corner in ["TL", "TR", "BR", "BL"]
]
# cols += [f"NEON_MARKER_{n}" for n in range(num_neon_markers)]
cols += ["RMAST", "LMAST", "RTRAG", "LTRAG"]

marker_df = pd.DataFrame()

marker_df["timestamps [ns]"] = time_qtm_in_neon

for marker in cols:
    index = marker_indices[marker]
    marker_pos = marker_positions[index].squeeze()

    if marker in ["RMAST", "LMAST", "RTRAG", "LTRAG"]:
        if marker == "RMAST":
            marker = "NEON_MARKER_1"
        elif marker == "LMAST":
            marker = "NEON_MARKER_2"
        elif marker == "RTRAG":
            marker = "NEON_MARKER_3"
        elif marker == "LTRAG":
            marker = "NEON_MARKER_4"

    marker_df[f"{marker}_X"] = list(marker_pos[0, good_rtrag_indices])
    marker_df[f"{marker}_Y"] = list(marker_pos[1, good_rtrag_indices])
    marker_df[f"{marker}_Z"] = list(marker_pos[2, good_rtrag_indices])

# export csv

marker_df.to_csv(output_path)
