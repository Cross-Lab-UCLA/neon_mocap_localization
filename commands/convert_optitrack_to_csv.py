import argparse

import numpy as np
import pandas as pd
import pyxdf  # type: ignore

import pupil_labs.neon_recording as plnr
from pupil_labs.neon_mocap_localization.cloud_recording import CloudRecording
from pupil_labs.neon_mocap_localization.load_optitrack_data import Take

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
parser.add_argument(
    "-t",
    "--trial_number",
    help="The trial number, used to align the start time of the OptiTrack data with the Neon data. If not provided, will use the first trial found in the xdf events.",
    required=False,
)

args = vars(parser.parse_args())

# load data

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

# EMMA'S CHANGE: commented out gaze stream search because some recordings did not have it (e.g. sub-PDIG02 calibration XDF) 
# xdf_gaze_data was found here but never used afterward (timing uses the Neon Events and Motive streams)

# xdf_gaze_data_idx = 0
# xdf_gaze_data = None
# while True:
#     if "Neon Companion_Neon Gaze" in xdf_data[0][xdf_gaze_data_idx]["info"]["name"]:
#         xdf_gaze_data = xdf_data[0][xdf_gaze_data_idx]
#         break
#     else:
#         xdf_gaze_data_idx += 1

# if xdf_gaze_data_idx > len(xdf_data[0]) or xdf_gaze_data is None:
    # TODO: raise error
#     pass

# since mocap starts after neon and ends before it, we can just shift the base
# mocap relative timestamps by that amount
mocap_start_labels = xdf_mocap_event_data["time_series"]

motive_start_indices = []
for i, events in enumerate(xdf_mocap_event_data["time_series"]):
    if 'MotiveStart' in events:  # only select label ['MotiveStart']
        # EMMA'S CHANGE: skip takes that are less than 1 s long (stop event < 1 s after start)
        ts = xdf_mocap_event_data["time_stamps"]
        nxt = xdf_mocap_event_data["time_series"][i + 1] if i + 1 < len(ts) else []
        if 'MotiveStop' in nxt and ts[i + 1] - ts[i] < 1.0:
            print(f"Skipping very short take at event {i} ({ts[i + 1] - ts[i]:.3f} s)")
            continue
        motive_start_indices.append(i)

print(f"Total number of MotiveStart trials found: {len(motive_start_indices)}")

if len(motive_start_indices) == 0:
    raise ValueError("No MotiveStart event found in mocap data!")

if args["trial_number"] is not None:
    trial_number = int(args["trial_number"])
    # EMMA'S CHANGE: fix trial alignment (previous code used start_idx = trial_number - 1 
    # which meant every trial after trial 1 was misaligned)
    # now start_idx uses the corresponding position in motive_start_indices list (contains all MotiveStart events)
    pos = trial_number - 1
    
    if pos >= len(motive_start_indices):
        raise ValueError(f"Trial {trial_number} exceeds {len(motive_start_indices)} trials found")
    start_idx = motive_start_indices[pos]  # event index of this trial's MotiveStart
    print(f"\nAligning to trial number {trial_number} (event index {start_idx})") 
else:
    start_idx = motive_start_indices[0]
    print(f"\nAligning to first MotiveStart trial (index {start_idx})")
    
opti_in_neon_offset = (
    xdf_mocap_event_data["time_stamps"][start_idx] - xdf_event_data["time_stamps"][0]
)

print("Alignment:")
print(f"Timeline offset: {opti_in_neon_offset:.3f} seconds")
print(f"Mocap start time:  {xdf_mocap_event_data['time_stamps'][start_idx]:.3f}s")
print(f"Neon start time:    {xdf_event_data['time_stamps'][0]:.3f}s")

# write the data out, synced & in a format compatible with the main localization
# script
output_df = pd.DataFrame({})

marker_time = opti_data.markers[next(iter(opti_data.markers.keys()))].times
output_df["timestamp [ns]"] = ((marker_time + opti_in_neon_offset) * 1e9).astype(
    np.int64
) + neon_rec.info["start_time"]  # type: ignore

for marker in opti_data.markers:
    # if there are missing frames, insert NaNs for those frames
    positions = opti_data.markers[marker].positions
    n_frames  = len(positions)
    marker_positions = np.full((n_frames, 3), np.nan, dtype=float)
    for i, p in enumerate(positions):
        if p is None:
            continue
        
        if not hasattr(p, "__len__"):
            continue

        L = min(len(p), 3)
        for j in range(L):
            marker_positions[i, j] = p[j]
    
    #marker_positions = np.array(opti_data.markers[marker].positions) 
    
    # save to output
    output_df[f"{marker}_X"] = marker_positions[:, 0]
    output_df[f"{marker}_Y"] = marker_positions[:, 1]
    output_df[f"{marker}_Z"] = marker_positions[:, 2]

output_df.to_csv(args["output_path"])
