import argparse

import numpy as np

from load_optitrack_data import Take

parser = argparse.ArgumentParser(
    description="Determines relative position of Neon scene camera in MoCap coordinate system"
)
parser.add_argument(
    "-mp",
    "--mocap_path",
    help="The path to the OptiTrack data",
    required=True,
)
parser.add_argument(
    "-nm", "--neon_markers", help="The number of markers on the wearers head", default=6
)

args = vars(parser.parse_args())

mocap_path = args["mocap_path"]
num_neon_markers = args["neon_markers"]

take = Take()
data = take.readCSV(mocap_path)

# extract neon's markers from optitrack data

neon_marker_positions = np.zeros((3, num_neon_markers))

for i in range(num_neon_markers):
    neon_marker_positions[:, i] = data.rigid_bodies[
        f"NEON_MARKERS:Marker 00{i + 1}"
    ].positions

# extract display markers from mocap data

apriltag_positions = np.zeros((4, 4, 3))

for tc in range(4):
    for mc in range(4):
        marker_num = tc * 4 + mc
        if marker_num > 9:
            apriltag_positions[tc, mc, :] = data.rigid_bodies[
                f"APRILTAG_MARKERS:Marker 0{marker_num + 1}"
            ].positions
        else:
            apriltag_positions[tc, mc, :] = data.rigid_bodies[
                f"APRILTAG_MARKERS:Marker 00{marker_num + 1}"
            ].positions
