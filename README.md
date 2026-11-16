# Neon Localization in MoCap Coordinate Systems

This repository provides a toolset for calculating the position and orientation (pose) of the Pupil Labs Neon scene camera within a Motion Capture (MoCap) coordinate system.

By combining these data streams during post-processing, users can generate 3D gaze vectors relative to the MoCap volume.

**Compatibility:**

- **Frames:** _Every Move You Make_, _I Can Track Clearly Now_, and custom frames with IR markers.
- **Headwear:** Custom markers placed directly on the head, or on a well-fitting cap/hat.

### Important: Validation

> [!TIP]
> Note: It is strongly recommended to pilot the complete workflow (Data Collection through to Data Processing) using test data prior to experimental data collection. Validating the pipeline ensures that hardware positioning, marker visibility, and time synchronization protocols are correctly configured before subject recruitment begins.

## Time Synchronization

Before proceeding to data collection, it is critical to establish a proper time synchronization protocol. Select the method appropriate for the hardware in use.

- **General Rule:** Start Neon recording **first**, then MoCap. Stop MoCap **first**, then Neon last.
- **Lab Streaming Layer (LSL):** Start LabRecorder **first**, then start/stop the Neon & MoCap streams according to the **General Rule**. Stop LabRecorder **last**.

**Specific Vendor Instructions:**

- **Qualisys / OptiTrack:** Use the **Lab Streaming Layer (LSL)** method. Capture Neon's LSL Gaze Stream and the MoCap LSL streams. If using Optitrack, make sure Motive's start/stop events are recorded via LSL.
- **Vicon:** Use a Vicon LockBox with an Arduino (or any suitable microcontroller or SBC, such as a Raspberry Pi):
  - The Vicon LockBox puts out regular TTL sync pulses. For example, you can use [the Duration.gpo configuration](https://vicon-help.atlassian.net/wiki/spaces/Connect/pages/1611203/Synchronization+output+GPO+configuration+files).
  - These TTL pulses can be received by the Arduino/SBC and timestamped, making sure to
    account for transmission delay. The delay can be measured with [Vicon's DataStream SDK](https://www.vicon.com/software/datastream-sdk/).
  - These can then be forwarded to a recording PC, which receives them via
    [pyserial](https://pyserial.readthedocs.io/en/latest/pyserial.html).
  - The recording PC then converts these to Neon [Events](https://docs.pupil-labs.com/neon/data-collection/events/) and sends them to Neon via the [Real-time API](https://pupil-labs.github.io/pl-realtime-api/dev/methods/simple/remote-control/#save-events). Make sure to apply [clock-offset correction](https://docs.pupil-labs.com/neon/data-collection/time-synchronization/#improving-synchronization-further), as shown with [example Python code here](https://pupil-labs.github.io/pl-realtime-api/dev/methods/simple/remote-control/#with-manual-clock-offset-correction).
  - If using the Duration.gpo configuration file, then make sure to update the Vicon start/end Event names in the `config.json` file accordingly (see the Workflows below for details).

If you use one of these methods, then your data will be compatible with our sync & conversion scripts described below.

## Workflow Options

We provide two different workflows for localizing Neon in MoCap coordinates:

- **Workflow A - Assume standard configuration:** This workflow assumes that you can sacrifice some accuracy for ease of use. It depends on using either the _Every Move You Make_ or _I Can Track Clearly Now_ frames.
- **Workflow B - Precise, person-specific mount localization:** This workflow is for research scenarios that seek as much accuracy as possible or for custom frames/mounts/headwear.

### Workflow A - Standard Configuration Overview

<details>
  <summary>View instructions</summary>

This procedure consists of two phases:

1. **Phase 1 - Data Collection:** Wear Neon with the standardized IR marker configuration while recording.
2. **Phase 2 - Data Processing:** Time sync the MoCap and Neon data. Then, apply the alignment script.

### Phase 1: Data Collection

#### 1. Hardware Requirements

This procedure requires the _Every Move You Make_ or _I Can Track Clearly Now_ frames.

Before recording, you must place the IR markers on the frames in the following standard configuration:

- Left Side (when wearing):
  - Top Marker - 5cm stick
  - Middle Marker - 6cm stick
  - Bottom Marker - 5cm stick
- Right Side (when wearing):
  - Top Marker - 6cm stick
  - Middle Marker - 5cm stick
  - Bottom Marker - 6cm stick

If you use any other configuration, then the provided script will produce inaccurate
results. If you need a different configuration, then see Workflow B.

#### 2. Recording Experimental Trials

Experimental trials may now proceed.

### Phase 2: Data Processing

Processing is performed after data collection is complete.

#### 1. Installation

Create a Python virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

Then, install this package into that virtual environment:

```bash
pip install -e .
```

#### 2. Data Preparation

Sync and convert the Motion Capture data to the required CSV format before localization.
Make sure to provide the arguments specified in the respective command's help display.

##### For Qualisys

```bash
python ./commands/convert_qualisys_to_csv.py -h
```

##### For OptiTrack

```bash
python ./commands/convert_optitrack_to_csv.py -h
```

##### For Vicon

```bash
python ./commands/convert_vicon_to_csv.py -h
```

#### 3. Configuration (`config.json`)

The `config.json` file controls the localization parameters. Ensure these match the physical setup. See the example in `examples/config_standard.json` for reference.

| Key                            | Type    | Description                                                                                                                                                                                           |
| ------------------------------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `qualisys_reference_marker`    | String  | The label of a clearly detected marker used for Qualisys LSL time sync. Leave as the empty string, "", if you do not use a Qualisys device.                                                           |
| `qualisys_frame_rate`          | Int     | The frame rate of the Qualisys recording.                                                                                                                                                             |
| `vicon_start_event_name`       | String  | The name of the Neon Event that corresponds to the **start** of a Vicon recording when using the Duration.gpo configuration.                                                                          |
| `vicon_end_event_name`         | String  | The name of the Neon Event that corresponds to the **end** of a Vicon recording when using the Duration.gpo configuration.                                                                            |
| `neutral_heading`              | Object  | An array of 3 floats specifying the neutral heading vector in your MoCap volume. That is, the vector perpendicular to gravity, parallel with the ground plane, and pointing in the forward direction. |
| `flip_gaze_x`                  | Boolean | If the MoCap coordinate system has X positive to the left, set this to true.                                                                                                                          |
| `flip_gaze_y`                  | Boolean | If the MoCap coordinate system has Y positive downwards, set this to true.                                                                                                                            |
| `flip_gaze_z`                  | Boolean | If the MoCap coordinate system has Z positive backwards, set this to true.                                                                                                                            |
| `mocap_unit_conversion_factor` | Float   | Internally, the scripts expect distances in meters. If your MoCap system does not record in meters, then use this option to scale it appropriately (default: `0.001`).                                |
| `neon_marker_labels`           | Object  | A map containing the labels assigned to the headset markers in the MoCap software. The keys of the map must be left untouched (e.g., "Top Left", "Right Middle"); only change the values.             |

</details>

### Workflow B - Precise Mount Localization Overview

<details>
  <summary>View instructions</summary>
This procedure consists of two phases:

1. **Phase 1 - Data Collection:** Recording the necessary calibration sequences and experimental trials.
2. **Phase 2 - Data Processing:** Calculating the transformation matrix and applying it to experimental data.

### Phase 1: Data Collection

#### 1. Hardware Requirements

To perform localization, the following elements must be present in the MoCap volume:

- **The Wearer:** The participant wearing Neon with attached IR reflective markers.
- **Calibration Board:** A rigid, flat board containing:
  - **Four AprilTags** from [the Tag36h11 family](https://raw.githubusercontent.com/pupil-labs/pupil-helpers/refs/heads/master/markers_stickersheet/tag36h11_full.pdf); IDs 0-3 are recommended for simplicity.
  - **Four IR Markers** centered precisely on the outermost corners of the AprilTags. That is, one IR marker on the top left corner of the top left tag, one on the top right corner of the top right tag, one on the bottom right corner of the bottom right tag, and one on the bottom left corner of the bottom left tag.
  - **Orientation:** The AprilTags must be upright (ID text at the bottom and legible).

#### 2. Coordinate System Alignment

By default, the scripts assume the following MoCap configuration based on a calibration bar placed flat on the floor/table (usually when setting up the motion capture volume)

- **X-Axis:** Points Right (when standing with outstretched arms).
- **Y-Axis:** Points Up (opposite gravity).
- **Z-Axis:** Points Forward (away from the body).

> [!TIP]
> Note: If a different convention is used, the `T_mocap_to_apriltag` matrix in `config.json` must be modified. It will most likely need to be a [permutation matrix](https://en.wikipedia.org/wiki/Permutation_matrix). Note that Neon follows OpenCV conventions (see the [Neon 3D eye pose diagram](https://docs.pupil-labs.com/neon/data-collection/data-streams/#_3d-eye-poses)).

#### 3. Recording the Calibration Sequence

A dedicated recording is required to compute the transformation matrix.

1. **Position:** Place the calibration board approximately at arm's length. The board should be no further than ~0.7m distance from the participant's head, regardless of whether it is sitting at waist height or higher/lower.
2. **Orientation**: The calibration board must be placed right side up. That is, the ID
   text of each printed AprilTag should be legibly oriented. It can help if the board is
   a bit tilted backward with respect to the line of sight.
3. **Procedure:** The participant should gaze at the the board for a recording of ~15-20 seconds total, while keeping their head still.
4. **Visibility:** Ensure the MoCap cameras detect all markers (frame and board) and that the Neon scene camera detects the AprilTags for the duration of the recording.

#### 4. Recording Experimental Trials

Once the calibration sequence is complete, experimental trials may proceed.

- The calibration board is not required for these trials.
- The participant must not move or remove the Neon frame (or markers) between the calibration sequence and the experimental trials.

### Phase 2: Data Processing

Processing is performed after data collection is complete.

#### 1. Installation

Create a Python virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

Then, install this package into that virtual environment:

```bash
pip install -e .
```

#### 2. Data Preparation

Sync and convert the Motion Capture data to the required CSV format before localization.
Make sure to provide the arguments specified in the respective command's help display.

##### For Qualisys

```bash
python ./commands/convert_qualisys_to_csv.py -h
```

##### For OptiTrack

```bash
python ./commands/convert_optitrack_to_csv.py -h
```

##### For Vicon

```bash
python ./commands/convert_vicon_to_csv.py -h
```

#### 3. Configuration (`config.json`)

The `config.json` file controls the localization parameters. Ensure these match the physical setup. See the example in `examples/config_precise.json` for reference. The parameters necessary for a precise calibration are detailed in the table below:

| Key                                 | Type    | Description                                                                                                                                                                                                                                                                           |
| ----------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `qualisys_reference_marker`         | String  | The label of a clearly detected marker used for Qualisys LSL time sync. Leave as the empty string, "", if you do not use a Qualisys device.                                                                                                                                           |
| `qualisys_frame_rate`               | Int     | The frame rate of the Qualisys recording.                                                                                                                                                                                                                                             |
| `vicon_start_event_name`            | String  | The name of the Neon Event that corresponds to the **start** of a Vicon recording when using the Duration.gpo configuration.                                                                                                                                                          |
| `vicon_end_event_name`              | String  | The name of the Neon Event that corresponds to the **end** of a Vicon recording when using the Duration.gpo configuration.                                                                                                                                                            |
| `neutral_heading`                   | Object  | An array of 3 floats specifying the neutral heading vector in your MoCap volume. That is, the vector perpendicular to gravity, parallel with the ground plane, and pointing in the forward direction.                                                                                 |
| `T_mocap_to_apriltag`               | Matrix  | Transformation matrix aligning the local coordinate system of the calibration board in MoCap space with the local coordinate system of the calibration board in Neon scene camera space.                                                                                              |
| `flip_gaze_x`                       | Boolean | If the MoCap coordinate system has X positive to the left, set this to true.                                                                                                                                                                                                          |
| `flip_gaze_y`                       | Boolean | If the MoCap coordinate system has Y positive downwards, set this to true.                                                                                                                                                                                                            |
| `flip_gaze_z`                       | Boolean | If the MoCap coordinate system has Z positive backwards, set this to true.                                                                                                                                                                                                            |
| `apriltag_black_side_length`        | Float   | The length of one black side of a printed AprilTag (in **meters**).                                                                                                                                                                                                                   |
| `ir_marker_radius`                  | Float   | The radius of the physical IR markers (in **meters**; default: `0.006`).                                                                                                                                                                                                              |
| `apriltag_pattern_width`            | Float   | The distance from the top left corner of the top left AprilTag to the top right corner of the top right AprilTag (in **meters**).                                                                                                                                                     |
| `apriltag_pattern_height`           | Float   | The distance from the top left corner of the top left AprilTag to the bottom left corner of the bottom left AprilTag (in **meters**).                                                                                                                                                 |
| `mocap_unit_conversion_factor`      | Float   | Internally, the scripts expect distances in meters. If your MoCap system does not record in meters, then use this option to scale it appropriately (default: `0.001`).                                                                                                                |
| `neon_marker_labels`                | Array   | The labels assigned to the headset markers in the MoCap software.                                                                                                                                                                                                                     |
| `apriltag_marker_labels`            | Object  | A map holding the labels assigned to the calibration board markers. The keys of the map must be "Top Left", "Top Right", "Bottom Right", and "Bottom Left" and should of course correspond to the IR markers in those positions.                                                      |
| `apriltags_to_use`                  | Array   | List of AprilTag IDs used on your board (e.g., `["0", "1", "2", "3"]`).                                                                                                                                                                                                               |
| `apriltag_corner_local_coordinates` | Object  | The local (X,Y) coordinates of the 16 AprilTag corners (see **Local Corner Measurements** below; default is meters, but other units are acceptable, see `corner_unit_conversion`). The coordinates are saved as array values in a map whose keys are the IDs from `apriltags_to_use`. |
| `corner_unit_conversion`            | Float   | Multiplier if your local coordinates are not in meters (default: `1.0`). See **Local Corner Measurements** below.                                                                                                                                                                     |

#### 4. Step A: Compute Calibration

Run the below script using the **Calibration Sequence** (from Phase 1, Step 3) to generate the pose file.

Use the following method to establish the board's position:

**Local Corner Measurements**

- Manually measure the 16 corners of the 4 AprilTags:
  - **Origin (0,0):** Top-Left corner of the Top-Left tag.
  - **Axes:** X positive to the right; Y positive down.
  - **Order:** List the corner measurements in the `apriltag_corner_local_coordinates` field of `config.json` in the following order: **Bottom Left, Bottom Right, Top Right, Top Left**.

```bash
python ./commands/mocap_compute_alignment.py -r [Neon_Folder] -m [MoCap_CSV] -c config.json
```

#### 4. Step B: Apply to Experimental Data

Use the `apply_alignment.py` script to apply the transformation matrix generated in Step 4.A to the **Experimental Trials**. This generates the final CSV file with gaze data in MoCap space.

_(Refer to the script's help arguments, `python apply_alignment.py -h`, for instructions on applying a saved transformation to new files.)_

</details>

## Visualization

We have included a Python script (`blender_plugin/mocap_visualizer.py`) that will add a
small plugin to a running Blender instance. Simply go to the Scripting tab in Blender,
make a new Text file, and copy the contents in. Press the `Play` button on the script
and you will have a new entry in the viewport Sidebar (known as the "N menu"). Fill in
the values and press the `Animate` button to have your MoCap and gaze data added to
Blender animation tracks. You can then playback your data in the Blender 3D viewport or
render it out as a video.

## Troubleshooting

- **Script not proceeding:**

  The `mocap_compute_alignment.py` script displays diagnostic plots (e.g., time sync offset) during execution. The plot window must be **closed** manually for the script to proceed to the next calculation step.

- **Time Sync Drift:**

  If gaze alignment appears to drift over time, verify that the correct conversion script was used in Phase 2, Step 2.

- **Inverted Axes:**

  If gaze appears mirrored, verify the `T_mocap_to_apriltag` matrix in `config.json` and ensure consistency with OpenCV coordinate conventions.
