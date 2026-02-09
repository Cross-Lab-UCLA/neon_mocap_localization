import argparse

import numpy as np
import pandas as pd
from ezc3d import c3d  #  type: ignore

import pupil_labs.neon_recording as plnr

parser = argparse.ArgumentParser(
    description="Determines relative position of Neon scene camera in MoCap \
coordinate system"
)

parser.add_argument(
    "-n",
    "--neon_rec_path",
    help="The folder with the Neon recording data",
    required=True,
)
parser.add_argument(
    "-c", "--c3d_path", help="The path to the Qualisys data (C3D file)", required=True
)
parser.add_argument(
    "-o",
    "--output_path",
    help="The path, in which to save the converted marker data",
    required=True,
)

args = vars(parser.parse_args())

neon_rec = plnr.open(args["neon_rec_path"])
c3d_data = c3d(args["c3d_path"])

c3d_frate = c3d_data["header"]["points"]["frame_rate"]
c3d_points = c3d_data["data"]["points"]

# vicon assumes perfectly constant sample rate
nframes = c3d_points.shape[2]
duration_s = nframes / c3d_frate

marker_names = c3d_data["parameters"]["POINT"]["LABELS"]["value"]

output_df = pd.DataFrame({})

output_df["timestamp [ns]"] = (
    np.arange(0, duration_s, step=1.0 / c3d_frate) * 1e9 + neon_rec.scene.time[0]
)

for idx, marker in enumerate(marker_names):
    marker_positions = c3d_points[:, idx, :].squeeze()

    output_df[f"{marker}_X"] = marker_positions[0, :]
    output_df[f"{marker}_Y"] = marker_positions[1, :]
    output_df[f"{marker}_Z"] = marker_positions[2, :]

output_df.to_csv(args["output_path"])
