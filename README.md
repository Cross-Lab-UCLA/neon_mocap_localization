# Localize Neon in Motion Capture Coordinate Systems

The scripts in this repository calculate the position and orientation (i.e., the pose) of Neon's scene camera 
in a Motion Capture (MoCap) coordinate system, thereby allowing you to accurately
represent gaze data in that space.

The `localize_neon_in_mocap.py` file and associated scripts are meant for the
[Every move you make](https://pupil-labs.com/products/neon/shop#every-move-you-make)
and the [I can track clearly now](https://pupil-labs.com/products/neon/shop#i-can-track-clearly-now)
frames, as well as [any custom frames](https://docs.pupil-labs.com/neon/hardware/make-your-own-frame/)
that carry Motion Capture markers. You could also attach IR markers to the head of the
participant or a secure hat/helment that they are wearing.

Before getting into details, be sure to make a fresh Python virtual environment and
install the necessary dependencies:

```
pip install -r requirements.txt
```

The scripts assume that you have configured your MoCap coordinate system as
follows:

- The calibration bar is placed with its arms, flat on the floor or a
  table surface, as the X and Z axes.
  - If you are standing with outstretched arms aligned with the X
    axis, then the X axis points to the right and the Z axis points
    forwards, away from your body.
- The Y axis is up, opposite gravity.

If this is not the case, then be sure to modify the
`T_neon_to_mocap` variable in the `config.json`
file accordingly. It is a matrix that specifies how to transform from Neon's coordinate system
contenions to the conventions of your MoCap system. For reference, Neon's scene camera coordinate system follows OpenCV conventions and is diagrammed [here](https://docs.pupil-labs.com/neon/data-collection/data-streams/#_3d-eye-poses).

The localization script requires the following two items:

1. A Neon Recording that was taken at the same time as the MoCap recording. (Notes about time sync are below.)
2. A `marker_positions.csv` file, with MoCap data captured during a
  simultaneous Neon recording, where the wearer is looking at
  AprilTags on a rigid, flat surface.
    - The AprilTags _must_ be accompanied by at least four MoCap IR markers that are
      also positioned on the same flat surface. Ideally, place each
      marker as precisely as possible on the corner of a black AprilTag corner.
    - You typically only need four AprilTags, arranged in a square/rectangular pattern.
    - We recommend using the first four AprilTags [from this document](https://github.com/pupil-labs/pupil-helpers/blob/master/markers_stickersheet/tag36h11_full.pdf?raw=True). Make sure that
    they are always oriented such that the line of text with the ID is upright.
    - If you use different AprilTags from the `tag36h11` family, then modify the
    `apriltags_to_use` entry in the `config.json` file to their IDs.

You can label the Neon and AprilTag IR markers as you wish. Just be sure to add their
labels to the `neon_marker_labels` and `apriltag_marker_labels` arrays in the
`config.json` file.

In the `config.json` file, you should also enter the following:

- The width of one black edge of a printed AprilTag, in the `apriltag_black_border_width` entry. It
must be entered in meters.
- The radius of your IR markers, in the `ir_marker_radius` entry. It should also be entered in meters.

The `localize_neon_in_mocap.py` script then offers two methods for localizing Neon in the MoCap space:

1. __Using screen-mapped gaze:__ The AprilTags can be used to define a Surface and gaze data can be
mapped to this Surface. This creates many 2D-3D correspondences to determine Neon's pose relative
to your AprilTag board. You can use either the [Marker Mapper](https://docs.pupil-labs.com/neon/pupil-cloud/enrichments/marker-mapper/) of Pupil Cloud or the [Surface Tracker](https://docs.pupil-labs.com/neon/neon-player/surface-tracker/)
of Neon Player to map gaze to the Surface. Simply instruct your participants to look around the center
of the board and at each AprilTag while the recording takes place. The resulting CSV file should then be passed
to the `TODO` argument of the `localize_neon_in_mocap.py` script.
2. __Using local AprilTag corner measurements:__ An alternate way to provide 2D-3D correspondences to
the algorithm is to measure the positions of the AprilTag corners in the local XY coordinate system of
the flat AprilTag board. Simply take the top left corner of the top left AprilTag as the origin (i.e.,
[0, 0]). Then, using a ruler, measure the 15 other corners of the AprilTags relative to it, with X positive to the right
and Y positive downwards. You can then enter these points into the `apriltag_corner_local_coordinates` entry
of the `config.json` file. This entry is a JSON map from AprilTag IDs to an array of coordinates for the respective tag's
4 corners.
  - It is expected that the coordinates are entered in meters. If not, change the `corner_unit_conversion_factor`
  accordingly.

Lastly, it is expected that you have properly [time synced](https://docs.pupil-labs.com/neon/data-collection/time-synchronization/) Neon with your MoCap device. It is also easiest to start the Neon recording first,
then the MoCap recording second. At the end, stop the MoCap recording **first** and the Neon recording **last**.
The scripts are essentially written with this assumption in mind.

Note that if you are using a Qualisys system, then you need to follow additional steps for time sync:

- You need to make a [Lab Streaming Layer](https://labstreaminglayer.org/) recording, in addition
  to the standard MoCap and Neon recordings.
  - Once LSL is installed, start LabRecorder first and instruct it to
    collect data from [Neon's LSL Gaze Stream](https://docs.pupil-labs.com/neon/data-collection/lab-streaming-layer/),
    as well as [Qualisys's MoCap LSL streams](https://github.com/qualisys/qualisys_lsl_app). Thereafter, start a Neon
    recording and then start the MoCap recording last. At the end, stop the MoCap recording first, then the Neon recording,
    and lastly the LSL recording.
- Update the `qualisys_reference_marker` entry in the `config.json` file to the label
of a clearly detected marker in the MoCap recording. This will be used as a reference
point for time sync.
- Then, run `convert_qualisys_to_csv.py`, providing also the XDF file
  that was produced by LSL to the `TODO` argument. Use the `-h` flag to see all the required arguments and their descriptions.
  
Using LSL to sync data from an Optitrack system to Neon is also recommended. The provided
`convert_optitrack_to_csv.py` script assumes you have used LSL to record Neon's Gaze and Event
channels, as well as the start/stop Events sent by the Motive software. Similarly, use the `-h` flag to see all the required arguments and their descriptions.

These time sync scripts should be run *before* the `localize_neon_in_mocap.py` script.

Finally, to run the localization script, do:

```
python localize_neon_in_mocap.py -r [folder with Neon Native Recording Data] -m marker_positions.csv -c config.json
```

It will save a "Neon & Mocap calibration file" in
`neon_camera_pose_relative_to_markers.json`. It will also print out
some diagnostic data and display some plots. Simply close each plot to
continue to the next step until completion.

After you have run the localization script, you can also make use of
the provided `gaze_to_mocap` function in your own analysis pipelines, as follows:

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
either the MoCap data or the Neon data to be at the same sampling rate.
