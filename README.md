# Neon Localization in MoCap Coordinate Systems

This repository provides a toolset for calculating the position and orientation (pose) of the Pupil Labs Neon scene camera within a Motion Capture (MoCap) coordinate system.

By combining these data streams during post-processing, users can generate 3D gaze vectors relative to the MoCap volume.

**Compatibility:**

- **Frames:** _Every Move You Make_, _I Can Track Clearly Now_, and custom frames with IR markers.
- **Headwear:** Custom markers placed directly on the head, or on a well-fitting cap/hat.

---

### Important: Validation

> [!TIP]
> Note: It is strongly recommended to pilot the complete workflow (Data Collection through to Data Processing) using test data prior to experimental data collection. Validating the pipeline ensures that hardware positioning, marker visibility, and time synchronization protocols are correctly configured before subject recruitment begins.

---

## Workflow Overview

The procedure consists of two phases:

1. **Phase 1 - Data Collection:** Recording the necessary calibration sequences and experimental trials.
2. **Phase 2 - Data Processing:** Calculating the transformation matrix and applying it to experimental data.

---

## Phase 1: Data Collection

### 1. Hardware Requirements

To perform localization, the following elements must be present in the MoCap volume:

- **The Wearer:** The participant wearing the Neon frame with attached IR reflective markers.
- **Calibration Board:** A rigid, flat board containing:
  - A minimum of **four AprilTags** (Tag36h11 family; IDs 0-3 are recommended).
  - **IR Markers** placed precisely at the corners of the AprilTags.
  - **Orientation:** AprilTags must be upright (ID text at the bottom).

### 2. Coordinate System Alignment

By default, the scripts assume the following MoCap configuration based on a calibration bar placed flat on the floor/table (usually when setting up the motion capture volume)

- **X-Axis:** Points Right (when standing with outstretched arms).
- **Y-Axis:** Points Up (opposite gravity).
- **Z-Axis:** Points Forward (away from the body).

> [!TIP]
> Note: If a different convention is used, the `T_neon_to_mocap` matrix in `config.json` must be modified. Note that Neon follows OpenCV conventions (see the [Neon 3D eye pose diagram](https://docs.pupil-labs.com/neon/data-collection/data-streams/#_3d-eye-poses)).

### 3. Time Synchronization

Proper time synchronization is critical. Select the method appropriate for the hardware in use.

- **General Rule:** Start Neon recording **first**, then MoCap. Stop MoCap **first**, then Neon last.
- **Lab Streaming Layer (LSL):** Start LabRecorder **first**, then start/stop the Neon & MoCap streams according to the **General Rule**. Stop LabRecorder **last**.

**Specific Vendor Instructions:**

- **Qualisys / OptiTrack:** Use the **Lab Streaming Layer (LSL)** method. Capture Neon's LSL Gaze Stream and the MoCap LSL streams. If using Optitrack, make sure Motive's start/stop events are recorded via LSL.
- **Vicon:** Ensure Vicon and Neon are synchronised via standard [Pupil Labs Time Sync](https://www.google.com/search?q=https://docs.pupil-labs.com/core/software/pupil-capture/%23time-sync) protocols.

### 4. Recording the Calibration Sequence

A dedicated recording is required to compute the transformation matrix.

1. **Position:** Place the calibration board approximately at arm's length and at head height. The board should be no further than ~0.7m distance from the participant.
2. **Procedure:** The participant should gaze at the center of the board and at each AprilTag for ~15-20 seconds.
3. **Visibility:** Ensure the MoCap cameras detect all markers (frame and board) and that the Neon scene camera detects the AprilTags for the duration of the recording.

### 5. Recording Experimental Trials

Once the calibration sequence is complete, experimental trials may proceed.

- The calibration board is not required for these trials.
- The participant must not move or remove the Neon frame (or markers) between the calibration sequence and the experimental trials.

---

## Phase 2: Data Processing

Processing is performed after data collection is complete.

### 1. Installation

Create a Python virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Sync and convert the Motion Capture data to the required CSV format before localization. Use

#### For Qualisys

```bash
python convert_qualisys_to_csv.py -x [path_to_xdf] -h
```

#### For OptiTrack

```bash
python convert_optitrack_to_csv.py -h
```

#### For Vicon

```bash
python convert_vicon_to_csv.py -h
```

### 3. Configuration (`config.json`)

The `config.json` file controls the localization parameters. Ensure these match the physical setup.

| Key                            | Type   | Description                                                                                                                                                                                           |
| ------------------------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `apriltags_to_use`             | Array  | List of AprilTag IDs used on your board (e.g., `[0, 1, 2, 3]`).                                                                                                                                       |
| `neon_marker_labels`           | Array  | Labels assigned to the headset markers in the MoCap software.                                                                                                                                         |
| `apriltag_marker_labels`       | Array  | Labels assigned to the calibration board markers.                                                                                                                                                     |
| `apriltag_black_border_width`  | Float  | The width of one black edge of a printed AprilTag (in **meters**).                                                                                                                                    |
| `ir_marker_radius`             | Float  | The radius of the physical IR markers (in **meters**).                                                                                                                                                |
| `T_neon_to_mocap`              | Matrix | Transformation matrix aligning Neon's coordinate system with the MoCap system.                                                                                                                        |
| `qualisys_reference_marker`    | String | A clearly detected marker label used for Qualisys LSL time sync. Leave as the empty string, "", if you do not use a Qualisys device.                                                                  |
| `apriltag_corner_local_coords` | Object | If using the `Local Corner Measurement` method (see below), the local (X,Y) coordinates of the 16 AprilTag corners (default is meters, but other units are acceptable, see `corner_unit_conversion`). |
| `corner_unit_conversion`       | Float  | Multiplier if your local coordinates are not in meters (default: `1.0`).                                                                                                                              |

### 4. Step A: Compute Calibration

Run the main script using the **Calibration Sequence** (from Phase 1, Step 4) to generate the pose file.

There are two methods for establishing the board's position:

**Method 1: Screen-Mapped Gaze (Recommended)**

- Map gaze to the calibration "Surface" in Pupil Cloud or Neon Player.
- Pass the resulting CSV to the script.

```bash
python localize_neon_in_mocap.py -r [Neon_Folder] -m [MoCap_CSV] -c config.json --surface_gaze_path [surface_gaze.csv]
```

**Method 2: Local Corner Measurements**

- Manually measure the 16 corners of the 4 AprilTags.
- **Origin (0,0):** Top-Left corner of the Top-Left tag.
- **Axes:** X positive to the right; Y positive down.
- **Order:** List corners in `config.json` as: **Bottom Left, Bottom Right, Top Right, Top Left**.

```bash
python localize_neon_in_mocap.py -r [Neon_Folder] -m [MoCap_CSV] -c config.json
```

### 5. Step B: Apply to Experimental Data

Apply the transformation matrix generated in Step A to the **Experimental Trials**. This generates the final `gaze_in_mocap_space.csv` file.

_(Refer to the script help arguments `python localize_neon_in_mocap.py -h` for instructions on applying a saved transformation to new files)._

---

## Troubleshooting

- **Plots not appearing:**

  The script displays diagnostic plots (e.g., time sync offset) during execution. The plot window must be **closed** manually for the script to proceed to the next calculation step.

- **Time Sync Drift:**

  If gaze alignment appears to drift over time, verify that the correct conversion script was used in Phase 2, Step 2.

- **Inverted Axes:**

  If gaze appears mirrored, verify the `T_neon_to_mocap` matrix in `config.json` and ensure consistency with OpenCV coordinate conventions.
