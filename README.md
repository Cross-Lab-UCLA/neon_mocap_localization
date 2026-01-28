# Neon Localization in MoCap Coordinate Systems

This repository provides tools to calculate the position and orientation (**pose**) of the Pupil Labs Neon scene camera within a Motion Capture (MoCap) coordinate system. This allows for high-precision representation of gaze data in 3D space.

It is compatible with [Every Move You Make](https://pupil-labs.com/products/neon/shop#every-move-you-make), [I Can Track Clearly Now](https://pupil-labs.com/products/neon/shop#i-can-track-clearly-now), and [custom frames](https://docs.pupil-labs.com/neon/hardware/make-your-own-frame/) equipped with IR markers. It is also compatible with custom markers placed directly on the head or a well-fitting cap/hat.

## Setup & Prerequisites

### Environment Setup

Create a fresh Python virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

### Coordinate System Alignment

By default, the scripts assume the following MoCap configuration:

* **X & Z Axes:** Calibration bar placed flat on the floor/table.
* **X-Axis:** Points right (when standing with outstretched arms).
* **Z-Axis:** Points forward (away from the body).
* **Y-Axis:** Points up (opposite gravity).

> [!TIP]
> If your system uses a different convention, modify the `T_neon_to_mocap` matrix in `config.json`. Neon follows **OpenCV conventions**; see the [Neon 3D eye pose diagram](https://docs.pupil-labs.com/neon/data-collection/data-streams/#_3d-eye-poses) for reference.

---

## Making a Recording

To perform a successful localization, you must capture at least the following two items simultaneously:

1. **Neon Recording:** A standard recording captured via the Neon Companion app or Pupil Cloud.
2. **MoCap Data:** A recording containing the trajectory of IR markers on the participant's frame and a flat "AprilTag Board."

### Hardware Requirements

* **The Board:** At least four AprilTags (recommended: IDs 0-3 from the `tag36h11` family, as contained [in this document](https://github.com/pupil-labs/pupil-helpers/blob/master/markers_stickersheet/tag36h11_full.pdf?raw=True)) on a rigid, flat surface.
* **Markers:** Place at least four IR markers precisely on the corners of the AprilTags.
* **Orientation:** Ensure AprilTags are upright (text ID at the bottom).

If you use different AprilTags from the `tag36h11` family, then modify the `apriltags_to_use` entry in the `config.json` file to contain their IDs.

---

## Time Synchronization

Proper time sync is critical. General rule: **Start Neon recording first, then MoCap. Stop MoCap first, then Neon last.**

If you use the [Lab Streaming Layer](https://labstreaminglayer.org/), then make sure that a recording is started in LabRecorder before Neon and the MoCap are started, and LabRecorder is the last to be stopped.

### Vicon

* Ensure Vicon and Neon are synced via standard [Pupil Labs Time Sync](https://docs.pupil-labs.com/neon/data-collection/time-synchronization/) protocols.

### Qualisys

1. Use the Lab Streaming Layer (LSL).
2. Start **LabRecorder** first, capturing [Neon's LSL Gaze Stream](https://docs.pupil-labs.com/neon/data-collection/lab-streaming-layer/) and [Qualisys LSL streams](https://github.com/qualisys/qualisys_lsl_app).
3. Run the conversion script before localization:
```bash
python convert_qualisys_to_csv.py -x [path_to_xdf] -h
```

### Optitrack

1. Record Neon Gaze/Events and Motive start/stop events via LSL.
2. Run the conversion script:
```bash
python convert_optitrack_to_csv.py -h
```

---

## Configuration (`config.json`)

The `config.json` file controls the localization parameters.

| Key | Type | Description |
| --- | --- | --- |
| `apriltags_to_use` | Array | List of AprilTag IDs used on your board (e.g., `[0, 1, 2, 3]`). |
| `neon_marker_labels` | Array | The labels assigned to IR markers on the Neon headset in your MoCap software. |
| `apriltag_marker_labels` | Array | The labels assigned to IR markers on the AprilTag board. |
| `apriltag_black_border_width` | Float | The width of one black edge of a printed AprilTag (in **meters**). |
| `ir_marker_radius` | Float | The radius of your physical IR markers (in **meters**). |
| `T_neon_to_mocap` | Matrix | Transformation matrix to align Neon's coordinate system with your MoCap. |
| `qualisys_reference_marker` | String | A clearly detected marker label used for Qualisys LSL time sync. |
| `apriltag_corner_local_coords` | Object | Local (X,Y) coordinates of the 16 AprilTag corners (in meters). |
| `corner_unit_conversion` | Float | Multiplier if your local coordinates are not in meters (default: `1.0`). |

---

## Localization Workflow

Run the main script to generate the pose file:

```bash
python localize_neon_in_mocap.py -r [Neon_Folder] -m [MoCap_CSV] -c config.json
```

### Methods for 2D-3D Correspondence

The script supports two methods for determining pose:

1. **Screen-Mapped Gaze:** Simply instruct your participants to look around the center of the board and at each AprilTag while the recording takes place. Then, map gaze to a "Surface" in [Pupil Cloud](https://docs.pupil-labs.com/neon/pupil-cloud/enrichments/marker-mapper/) or [Neon Player](https://docs.pupil-labs.com/neon/neon-player/surface-tracker/). Pass the resulting CSV to `--surface_gaze_path`.
2. **Local Corner Measurements:** Manually measure the 16 corners of your 4 AprilTags. The origin (0,0) is the Top-Left of the Top-Left tag, with X positive to the right and Y positive down. List corners as: **Bottom Left, Bottom Right, Top Right, Top Left.**

---

## Troubleshooting

* **Plots not appearing:** The script displays diagnostic plots. You must close the current plot window for the script to proceed to the next calculation step.
* **Time Sync Drift:** Ensure you are using the conversion scripts (`convert_qualisys_to_csv.py` etc.) *before* running the localization script.
* **Inverted Axes:** If the gaze appears mirrored, double-check your `T_neon_to_mocap` matrix and OpenCV coordinate conventions.
