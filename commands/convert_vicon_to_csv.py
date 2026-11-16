import argparse
import json

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
    "-c3", "--c3d_path", help="The path to the Vicon data (C3D file)", required=True
)
parser.add_argument(
    "-c", "--config_path", help="The path to the config file", required=True
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

with open(args["config_path"]) as f:
    config = json.load(f)

c3d_frate = c3d_data["header"]["points"]["frame_rate"]
c3d_points = c3d_data["data"]["points"]

# parse sync events
events_ts = neon_rec.events.time
events_names = neon_rec.events.event
mocap_start_ts = None
mocap_end_ts = None
for i, name in enumerate(events_names):
    if name == config["vicon_start_event_name"]:
        mocap_start_neon_ts_ns = events_ts[i]
    elif name == config["vicon_end_event_name"]:
        mocap_end_neon_ts_ns = events_ts[i]


# vicon assumes perfectly constant sample rate
nframes = c3d_points.shape[2]
duration_s = nframes / c3d_frate

marker_names = c3d_data["parameters"]["POINT"]["LABELS"]["value"]

output_df = pd.DataFrame({})

marker_time = np.arange(
    mocap_start_neon_ts_ns,
    mocap_end_neon_ts_ns,
    step=int((1.0 / c3d_frate) * 1e9),
    dtype=np.int64,
)

if len(marker_time) - nframes == 1:
    marker_time = marker_time[:nframes]
elif len(marker_time) - nframes == -1:
    marker_time = np.append(marker_time, marker_time[-1] + int((1.0 / c3d_frate) * 1e9))
elif len(marker_time) == nframes:
    pass
else:
    raise ValueError(
        "The number of generated timestamps does not match the number of frames in the \
        C3D data. Please check the sync events and frame rate."
    )

output_df["timestamp [ns]"] = marker_time

for idx, marker in enumerate(marker_names):
    marker_positions = c3d_points[:, idx, :].squeeze()

    output_df[f"{marker}_X"] = marker_positions[0, :]
    output_df[f"{marker}_Y"] = marker_positions[1, :]
    output_df[f"{marker}_Z"] = marker_positions[2, :]

output_df.to_csv(args["output_path"])
