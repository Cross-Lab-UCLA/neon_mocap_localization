import json

import numpy as np
import pyxdf  #  type: ignore
import scipy.io as sio
from ezc3d import c3d  #  type: ignore

import pupil_labs.neon_recording as plnr


class QualisysRecording:
    def __init__(
        self,
        xdf_path,
        c3d_path,
        mat_path,
        neon_rec_path,
        config_path,
    ):
        self.xdf_path = xdf_path
        self.c3d_path = c3d_path
        self.mat_path = mat_path
        self.neon_rec_path = neon_rec_path
        self.config_path = config_path

        self.qtm_trim_begin = None
        self.qtm_trim_end = None

        self.neon_trim_begin = None
        self.neon_trim_end = None

        self.config = []

        self.aligned_qtm_ts = None
        self.aligned_gaze_ts = None

        self._load()

        self._process_qtm()
        self._obtain_qtm_reference_data()

        self._process_xdf()
        self._obtain_xdf_reference_data()

        self._obtain_neon_reference_data()

        self._process_c3d()
        self._obtain_c3d_reference_data()

    def _load(self):
        with open(self.config_path) as f:
            self.config = json.load(f)

        self.qtm_data = sio.loadmat(self.mat_path)
        self.xdf_data = pyxdf.load_xdf(self.xdf_path, dejitter_timestamps=False)
        self.c3d_data = c3d(self.c3d_path)

        self.neon_rec = plnr.open(self.neon_rec_path)

    def _process_c3d(self):
        self.c3d_points = self.c3d_data["data"]["points"]
        self.mocap_nframes = self.c3d_points.shape[2]
        self.mocap_frate = self.c3d_data["header"]["points"]["frame_rate"]

        self.mocap_timestamps = np.arange(self.mocap_nframes) / self.mocap_frate

    def _obtain_c3d_reference_data(self):
        pass

    def _process_qtm(self):
        # extract relevant marker positions from mocap data

        self.condition_name = list(self.qtm_data.keys())[-1]
        try:
            self.qtm_marker_positions = self.qtm_data[self.condition_name][0][0][5][0][
                0
            ][0][0][0][2]
        except Exception:
            self.qtm_marker_positions = self.qtm_data[self.condition_name][0][0][6][0][
                0
            ][0][0][0][2]

        self.nsamples = self.qtm_marker_positions.shape[2]

        # Indices of relevant markers in qtm_marker_positions array

        try:
            self.marker_names = self.qtm_data[self.condition_name][0][0][5][0][0][0][0][
                0
            ][1][0]
        except Exception:
            self.marker_names = self.qtm_data[self.condition_name][0][0][6][0][0][0][0][
                0
            ][1][0]

        self.marker_indices = {
            str(name[0]): idx for idx, name in enumerate(self.marker_names)
        }

    def _obtain_qtm_reference_data(self):
        self.qtm_reference_positions = self.qtm_marker_positions[
            self.marker_indices[self.config["qualisys_reference_marker"]]
        ].squeeze()

        if self.qtm_reference_positions.shape[1] == 4:
            self.qtm_reference_positions = self.qtm_reference_positions.T

        self.qtm_reference_duration = (
            self.qtm_reference_positions.shape[-1] / self.config["qualisys_frame_rate"]
        )
        self.qtm_reference_timestamps = np.arange(
            0, self.qtm_reference_duration, 1 / self.config["qualisys_frame_rate"]
        )

        self.qtm_trim_begin = int(self.qtm_trim_begin) if self.qtm_trim_begin else 0
        self.qtm_trim_end = (
            -int(self.qtm_trim_end + 1)
            if self.qtm_trim_end is not None
            else self.qtm_reference_positions.shape[1]
        )

        self.qtm_ts_for_alignment = np.arange(
            self.qtm_reference_timestamps[0], self.qtm_reference_timestamps[-1], 1 / 200
        )
        self.qtm_data_resampled = np.interp(
            self.qtm_ts_for_alignment,
            self.qtm_reference_timestamps,
            self.qtm_reference_positions[0, :].squeeze(),
        )
        self.qtm_data_for_alignment = self.qtm_data_resampled[
            self.qtm_trim_begin : self.qtm_trim_end
        ]

    def _process_xdf(self):
        self.qualisys_xdf_idx = np.nan
        self.neon_xdf_idx = np.nan
        for idx, channel in enumerate(self.xdf_data[0]):
            if channel["info"]["name"][0] == "Qualisys":
                self.qualisys_xdf_idx = idx
            elif channel["info"]["name"][0] == "Neon Companion_Neon Gaze":
                self.neon_xdf_idx = idx

        self.reference_xdf_data_idx = np.nan
        for idx, channel in enumerate(
            self.xdf_data[0][self.qualisys_xdf_idx]["info"]["desc"][0]["channels"][0][
                "channel"
            ]
        ):
            if self.config["qualisys_reference_marker"] in channel["label"][0]:
                self.reference_xdf_data_idx = idx
                break

    def _obtain_xdf_reference_data(self):
        self.xdf_reference_timestamps = self.xdf_data[0][self.qualisys_xdf_idx][
            "time_stamps"
        ]
        self.xdf_reference_positions = (
            self.xdf_data[0][self.qualisys_xdf_idx]["time_series"][
                :, self.reference_xdf_data_idx : self.reference_xdf_data_idx + 3
            ].squeeze()
            * 1000
        )  # convert to millimeters

        self.xdf_ts_for_alignment = np.arange(
            self.xdf_reference_timestamps[0], self.xdf_reference_timestamps[-1], 1 / 200
        )
        self.xdf_data_for_alignment = np.interp(
            self.xdf_ts_for_alignment,
            self.xdf_reference_timestamps,
            self.xdf_reference_positions[:, 0].squeeze(),
        )

        self.xdf_gaze_timestamps = self.xdf_data[0][self.neon_xdf_idx]["time_stamps"]
        self.xdf_reference_gaze = self.xdf_data[0][self.neon_xdf_idx]["time_series"][
            :, 0
        ]

        self.xdf_gaze_ts_for_alignment = np.arange(
            self.xdf_gaze_timestamps[0], self.xdf_gaze_timestamps[-1], 1 / 200
        )
        self.xdf_neon_data_for_alignment = np.interp(
            self.xdf_gaze_ts_for_alignment,
            self.xdf_gaze_timestamps,
            self.xdf_reference_gaze,
        )

    def _obtain_neon_reference_data(self):
        self.neon_reference_timestamps = self.neon_rec.gaze.time * 1e-9
        self.neon_reference_gaze = self.neon_rec.gaze.data["point_x"]

        self.neon_trim_begin = int(self.neon_trim_begin) if self.neon_trim_begin else 0
        self.neon_trim_end = (
            -int(self.neon_trim_end + 1)
            if self.neon_trim_end is not None
            else len(self.neon_reference_gaze)
        )

        self.neon_ts_for_alignment = np.arange(
            self.neon_reference_timestamps[0],
            self.neon_reference_timestamps[-1],
            1 / 200,
        )
        self.neon_gaze_resampled = np.interp(
            self.neon_ts_for_alignment,
            self.neon_reference_timestamps,
            self.neon_reference_gaze,
        )
        self.neon_gaze_for_alignment = self.neon_gaze_resampled[
            self.neon_trim_begin : self.neon_trim_end
        ]
