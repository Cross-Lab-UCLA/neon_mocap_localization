# Localize Neon in Optitrack Coordinate System

The `localize_neon_in_optitrack.py` file is meant for the [Every move you make](https://pupil-labs.com/products/neon/shop#every-move-you-make) and the [I can track clearly now](https://pupil-labs.com/products/neon/shop#i-can-track-clearly-now) frames, as well as [any custom frames](https://docs.pupil-labs.com/neon/hardware/make-your-own-frame/) that carry Motion Capture markers.

It calculates the position of Neon's scene camera in the Optitrack coordinate system.
It could be modified for other MoCap systems.

It assumes that you have configured your copy of Motiv as follows:

- The Tracking Calibration bar is placed with its arms, flat on the floor or a table surface, as the X and Y axes.
- The Z axis is up.

While running, it produces some plots to help see what it is doing.
Simply close each plot when you are done inspecting it for the script to proceed to the next step.

The script requires the following:

- An OptiTrack `Take` during a Neon recording, where the wearer is looking [at AprilTags on a flat surface](https://docs.pupil-labs.com/neon/pupil-cloud/enrichments/marker-mapper/#setup).
  - The AprilTags _must_ be accompanied by MoCap IR markers that are also positioned on the same flat surface.
- An image of the AprilTags, as taken with Neon's scene camera during said Optitrack `Take`.
- Neon's scene camera intrinsics data, as contained in [the `scene_camera.json` file](https://docs.pupil-labs.com/neon/data-collection/data-format/#scene-camera-json).

To run it, first install the requirements:

```
pip install -r requirements.txt
```

Then:

```
python localize_neon_in_optitrack.py --image apriltags.png --calib scene_camera.json --optitrack optitrack_take.csv
```

It will save a "Neon+OptiTrack calibration file" in `neon_camera_pose_relative_to_markers.json`. It will also print out some diagnostic data and display some plots. Simply close each plot to continue to the next step until completion.

After you have run the localization script, you can then make use of the `gaze_to_optitrack` function in the `map_gaze_to_optitrack.py` file, as follows:

```python
from pose import Pose
from map_gaze_to_optitrack import map_gaze_to_optitrack

optitrack_transform = []
with open("neon_camera_pose_relative_to_markers.json", "r") as f:
    optitrack_transform = json.load(f)
    neon_relative_pose = Pose(
        position=optitrack_transform["position"],
        rotation=optitrack_transform["rotation"],
    )

map_gaze_to_optitrack(
    neon_relative_pose,
    azimuth,
    elevation,
    avg_neon_marker_positions,
)
```

Azimuth & elevation are provided for each Neon gaze datum in [`gaze.csv`](https://docs.pupil-labs.com/neon/data-collection/data-format/#gaze-csv).

The value, `avg_neon_marker_positions`, needs to be calculated for each frame of an OptiTrack Take. It would be advised to interpolate either the OptiTrack data or the Neon data to be at the same sampling rate and then temporally sync them. You can use principles from [our Time Sync guide](https://docs.pupil-labs.com/neon/data-collection/time-synchronization/).
