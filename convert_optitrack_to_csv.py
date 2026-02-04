import argparse

import numpy as np
import pandas as pd
import pyxdf

import pupil_labs.neon_recording as plnr
from cloud_recording import CloudRecording
from load_optitrack_data import Take

parser = argparse.ArgumentParser(
    description="Determines relative position of Neon scene camera in MoCap \
coordinate system"
)

parser.add_argument(
    "-x", "--xdf_path", help="The XDF file with the LSL data", required=True
)
parser.add_argument(
    "-op", "--optitrack", help="The CSV file with the raw OptiTrack data", required=True
)
parser.add_argument(
    "-n",
    "--neon_rec_path",
    help="The folder with the Neon recording data",
    required=True,
)
parser.add_argument(
    "-o",
    "--output_path",
    help="The path, in which to save the converted marker data",
    required=True,
)

args = vars(parser.parse_args())

# load data

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
    except Exception as e:
        raise ValueError("Not a valid Neon data directory") from e

# load xdf

xdf_data = pyxdf.load_xdf(args["xdf_path"])

# load raw optitrack data

opti_data = Take().readCSV(args["optitrack"])

# find the neon events channel in xdf

xdf_events_idx = 0
xdf_event_data = None
while True:
    if "Neon Companion_Neon Events" in xdf_data[0][xdf_events_idx]["info"]["name"]:
        xdf_event_data = xdf_data[0][xdf_events_idx]
        break
    else:
        xdf_events_idx += 1

if xdf_events_idx > len(xdf_data[0]) or xdf_event_data is None:
    # TODO: raise error
    pass

# find the mocap events channel in xdf

xdf_mocap_events_idx = 0
xdf_mocap_event_data = None
while True:
    if "Motive" in xdf_data[0][xdf_mocap_events_idx]["info"]["name"]:
        xdf_mocap_event_data = xdf_data[0][xdf_mocap_events_idx]
        break
    else:
        xdf_mocap_events_idx += 1

if xdf_mocap_events_idx > len(xdf_data[0]) or xdf_mocap_event_data is None:
    # TODO: raise error
    pass

# find the gaze data channel in xdf

xdf_gaze_data_idx = 0
xdf_gaze_data = None
while True:
    if "Neon Companion_Neon Gaze" in xdf_data[0][xdf_gaze_data_idx]["info"]["name"]:
        xdf_gaze_data = xdf_data[0][xdf_gaze_data_idx]
        break
    else:
        xdf_gaze_data_idx += 1

if xdf_gaze_data_idx > len(xdf_data[0]) or xdf_gaze_data is None:
    # TODO: raise error
    pass

# since mocap starts after neon and ends before it, we can just shift the base
# mocap relative timestamps by that amount

opti_in_neon_offset = (
    xdf_mocap_event_data["time_stamps"][0] - xdf_event_data["time_stamps"][0]
)

# write the data out, synced & in a format compatible with the main localization
# script

output_df = pd.DataFrame({})

marker_time = opti_data.markers[next(iter(opti_data.markers.keys()))].times
output_df["timestamp [ns]"] = ((marker_time + opti_in_neon_offset) * 1e9).astype(
    np.int64
) + neon_rec.info["start_time"]

for marker in opti_data.markers:
    marker_positions = np.array(opti_data.markers[marker].positions)

    output_df[f"{marker}_X"] = marker_positions[:, 0]
    output_df[f"{marker}_Y"] = marker_positions[:, 1]
    output_df[f"{marker}_Z"] = marker_positions[:, 2]

output_df.to_csv(args["output_path"])
