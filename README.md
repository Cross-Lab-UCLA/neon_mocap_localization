# Localize Neon in Motion Capture Coordinate Systems

The `localize_neon_in_mocap.py` file and associated scripts are meant for the [Every move you make](https://pupil-labs.com/products/neon/shop#every-move-you-make) and the [I can track clearly now](https://pupil-labs.com/products/neon/shop#i-can-track-clearly-now) frames, as well as [any custom frames](https://docs.pupil-labs.com/neon/hardware/make-your-own-frame/) that carry Motion Capture markers.

It calculates the position of Neon's scene camera in a Motion Capture
(MoCap) coordinate system, thereby allowing you to accurately
represent gaze data in that space.

The scripts assume that you have configured your MoCap coordinate system as
follows:

- The calibration bar is placed with its arms, flat on the floor or a
  table surface, as the X and Y axes.
  - If you are standing with outstretched arms aligned with the X
    axis, then the X axis points to the right and the Y axis points
    forwards, away from your body.
- The Z axis is up, opposite gravity.

If this is not the case, then you can modify the
`R_apriltag_to_mocap` variable in the `localize_neon_in_mocap.py`
file.

The localization script requires the following two things:

1. A `marker_positions.csv` file, with MoCap data captured during a
  simultaneous Neon recording, where the wearer is looking [at
  AprilTags on a flat
  surface](https://docs.pupil-labs.com/neon/pupil-cloud/enrichments/marker-mapper/#setup).
  - The AprilTags _must_ be accompanied by four MoCap IR markers that are
    also positioned on the same flat surface. Ideally, place each
    marker just at the edge of the black AprilTag corners. 
  - You should use only four AprilTags and place them in a square pattern.
2. A Neon Recording that was taken at the same time as the MoCap recording.

Make sure to label the IR markers as follows:

- All markers for a given AprilTag should follow the format: `T1TL`,
  where `T1` means `tag 1` and `TL` means the `t`op `l`eft corner of
  that tag. The other labels should be as follows:
   - `T1TR` -> tag 1; top right
   - `T1BR` -> tag 1; bottom right
   - `T1BL` -> tag 1; bottom left
   - and similarly for Tags 2 (`T2`), 3 (`T3`), and 4 (`T4`).
- All markers for Neon should follow the format: `NEON_MARKER_x`,
  where `x` varies from `1` to as many markers as you used for
  tracking Neon.
  
Provided you followed these labelling conventions, you can use our
provided basic conversion scripts to take data from popular MoCap
distributors and convert it to a `marker_positions.csv` file.

It is expected that you have properly [time
synced](https://docs.pupil-labs.com/neon/data-collection/time-synchronization/)
Neon with your MoCap device. If you are using a Qualisys system, then
you need to take specific steps for time sync:

- You need to also make a [Lab Streaming Layer](https://labstreaminglayer.org/) recording, in addition
  to the standard MoCap and Neon recordings
  - Once LSL is installed, start LabRecorder and instruct it to
    collect data from [Neon's LSL Gaze
    Stream](https://docs.pupil-labs.com/neon/data-collection/lab-streaming-layer/),
    as well as Qualisys's MoCap streams.
- Then, run `convert_qualisys_to_csv.py`, providing also the XDF file
  that was produced by LSL.

Now, to run the localization script, first install the requirements:

```
pip install -r requirements.txt
```

Then, run:

```
python localize_neon_in_mocap.py -r [folder with Neon Native Recording Data] -m marker_positions.csv
```

It will save a "Neon+Mocap calibration file" in
`neon_camera_pose_relative_to_markers.json`. It will also print out
some diagnostic data and display some plots. Simply close each plot to
continue to the next step until completion.

After you have run the localization script, you can then make use of
the `gaze_to_mocap` function in the `map_gaze_to_mocap.py` file, as
follows:

```python
from pose import Pose
from map_gaze_to_mocap import map_gaze_to_mocap

mocap_transform = []
with open("neon_camera_pose_relative_to_markers.json", "r") as f:
    mocap_transform = json.load(f)
    neon_relative_pose = Pose(
        position=mocap_transform["position"],
        rotation=mocap_transform["rotation"],
    )

map_gaze_to_mocap(
    neon_relative_pose,
    azimuth,
    elevation,
    avg_neon_marker_positions,
)
```

Azimuth & elevation are provided for each Neon gaze datum in
[`gaze.csv`](https://docs.pupil-labs.com/neon/data-collection/data-format/#gaze-csv).

The value, `avg_neon_marker_positions`, needs to be calculated for
each frame of a MoCap recording. It would be advised to interpolate
either the MoCap data or the Neon data to be at the same sampling rate
and then temporally sync them.
