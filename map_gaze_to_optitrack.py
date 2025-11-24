import numpy as np
from scipy.spatial.transform import Rotation as R

# Load `neon_relative_pose` as follows:
#
# optitrack_transform = []
# with open("neon_camera_pose_relative_to_markers.json", "r") as f:
#     optitrack_transform = json.load(f)
#     neon_relative_pose = Pose(
#         position=optitrack_transform["position"],
#         rotation=optitrack_transform["rotation"],
#     )


# azimuth & elevation as provided by Neon's gaze.csv:
# https://docs.pupil-labs.com/neon/data-collection/data-format/#gaze-csv
def map_gaze_to_optitrack(
    neon_relative_pose,
    azimuth,
    elevation,
    avg_neon_marker_positions,
):
    base_gaze_vector = np.array([0, 0, 1.0])

    rot_z = R.from_euler("z", azimuth, degrees=True)
    gaze_vector = rot_z.apply(base_gaze_vector)

    rot_x = R.from_euler("x", elevation, degrees=True)
    gaze_vector = rot_x.apply(gaze_vector)

    gaze_vector = neon_relative_pose.rotation.apply(gaze_vector)

    gaze_in_optitrack = {
        "position": neon_relative_pose.position + avg_neon_marker_positions,
        "direction": gaze_vector,
    }

    return gaze_in_optitrack
