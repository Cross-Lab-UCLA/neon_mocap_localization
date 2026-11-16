import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from scipy.interpolate import interp1d

from pupil_labs.neon_mocap_localization import time_sync_utils
from pupil_labs.neon_mocap_localization.qualisys import data_loader


class MainWindow(QMainWindow):
    def __init__(self, qualisys_rec, mat_export_dir, xdf_export_dir):
        super().__init__()

        self.qualisys_rec = qualisys_rec
        self.mat_export_dir = mat_export_dir
        self.xdf_export_dir = xdf_export_dir

        self.setWindowTitle("Qualisys <-> Neon Synchronization Tool")

        # Window dimensions
        geometry = self.screen().availableGeometry()
        self.setFixedSize(geometry.width() * 0.8, geometry.height() * 0.49)

        self.main_layout = QHBoxLayout()
        self.central_widget = QWidget()

        self._prepare_qualisys_window()
        self._prepare_neon_window()
        self.prepare_export_column()

        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

    def _prepare_signal_window(
        self, prefix, trim_begin_slot, trim_end_slot, update_plot
    ):
        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvas(fig)
        axes = fig.add_subplot(111)
        toolbar = NavigationToolbar2QT(canvas, self)

        begin_slider = QSlider(Qt.Horizontal)
        begin_slider.setMinimum(0)
        begin_slider.setMaximum(5000)
        begin_slider.setValue(0)
        begin_slider.valueChanged.connect(trim_begin_slot)
        begin_label = QLabel("Trim begin: 0")

        end_slider = QSlider(Qt.Horizontal)
        end_slider.setMinimum(0)
        end_slider.setMaximum(5000)
        end_slider.setValue(0)
        end_slider.valueChanged.connect(trim_end_slot)
        end_label = QLabel("Trim end: 0")

        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        layout.addWidget(begin_label)
        layout.addWidget(begin_slider)
        layout.addWidget(end_label)
        layout.addWidget(end_slider)

        self.main_layout.addLayout(layout)

        setattr(self, f"{prefix}_fig", fig)
        setattr(self, f"{prefix}_canvas", canvas)
        setattr(self, f"{prefix}_axes", axes)
        setattr(self, f"{prefix}_toolbar", toolbar)
        setattr(self, f"{prefix}_begin_slider", begin_slider)
        setattr(self, f"{prefix}_begin_label", begin_label)
        setattr(self, f"{prefix}_end_slider", end_slider)
        setattr(self, f"{prefix}_end_label", end_label)
        setattr(self, f"{prefix}_layout", layout)

        update_plot()

    def _prepare_qualisys_window(self):
        self._prepare_signal_window(
            "qtm",
            self.update_qualisys_trim_begin,
            self.update_qualisys_trim_end,
            self.update_qualisys_plot,
        )

    @Slot(int)
    def update_qualisys_trim_begin(self, value):
        self.qtm_begin_label.setText(f"Trim begin: {value}")
        self.qualisys_rec.qtm_trim_begin = value

        self.update_qualisys_plot()

    @Slot(int)
    def update_qualisys_trim_end(self, value):
        self.qtm_end_label.setText(f"Trim end: {value}")
        self.qualisys_rec.qtm_trim_end = -(value + 1)

        self.update_qualisys_plot()

    def update_qualisys_plot(self):
        self.qtm_axes.clear()

        self.qtm_axes.set_title("Qualisys Data")
        self.qtm_axes.set_xlabel("LSL Time [s]")
        self.qtm_axes.set_ylabel("X position [mm]")

        self.qualisys_rec.qtm_data_for_alignment = self.qualisys_rec.qtm_data_resampled[
            self.qualisys_rec.qtm_trim_begin : self.qualisys_rec.qtm_trim_end
        ]

        offs = time_sync_utils.align_signals(
            self.qualisys_rec.qtm_data_for_alignment,
            self.qualisys_rec.xdf_data_for_alignment,
            self.qualisys_rec.xdf_ts_for_alignment,
        )

        self.qualisys_rec.aligned_qtm_ts = (
            self.qualisys_rec.qtm_reference_timestamps
            + self.qualisys_rec.xdf_ts_for_alignment[
                offs - self.qualisys_rec.qtm_trim_begin
            ]
        )

        self.qtm_axes.plot(
            self.qualisys_rec.xdf_reference_timestamps,
            self.qualisys_rec.xdf_reference_positions[:, 0].squeeze(),
        )
        self.qtm_axes.plot(
            self.qualisys_rec.aligned_qtm_ts,
            self.qualisys_rec.qtm_reference_positions[0, :].squeeze(),
        )

        # mid_idx = len(self.qualisys_rec.aligned_qtm_ts) // 2
        # mid_ts = self.qualisys_rec.aligned_qtm_ts[mid_idx]
        # self.qtm_axes.set_xlim(mid_ts - 0.5, mid_ts + 0.5)
        self.qtm_axes.set_ylim(
            np.min(self.qualisys_rec.qtm_data_resampled),
            np.max(self.qualisys_rec.qtm_data_resampled),
        )

        self.qtm_canvas.draw()

    def _prepare_neon_window(self):
        self._prepare_signal_window(
            "neon",
            self.update_neon_trim_begin,
            self.update_neon_trim_end,
            self.update_neon_plot,
        )

    @Slot(int)
    def update_neon_trim_begin(self, value):
        self.neon_begin_label.setText(f"Trim begin: {value}")
        self.qualisys_rec.neon_trim_begin = value

        self.update_neon_plot()

    @Slot(int)
    def update_neon_trim_end(self, value):
        self.neon_end_label.setText(f"Trim end: {value}")
        self.qualisys_rec.neon_trim_end = -(value + 1)

        self.update_neon_plot()

    def update_neon_plot(self):
        self.neon_axes.clear()

        self.neon_axes.set_title("Neon Gaze Data")
        self.neon_axes.set_xlabel("LSL Time [s]")
        self.neon_axes.set_ylabel("X position [px]")

        self.qualisys_rec.neon_gaze_for_alignment = (
            self.qualisys_rec.neon_gaze_resampled[
                self.qualisys_rec.neon_trim_begin : self.qualisys_rec.neon_trim_end
            ]
        )

        neon_offset = time_sync_utils.align_signals(
            self.qualisys_rec.neon_gaze_for_alignment,
            self.qualisys_rec.xdf_neon_data_for_alignment,
            self.qualisys_rec.xdf_gaze_ts_for_alignment,
        )

        self.qualisys_rec.aligned_gaze_ts = (
            self.qualisys_rec.neon_reference_timestamps
            - self.qualisys_rec.neon_reference_timestamps[0]
        ) + self.qualisys_rec.xdf_gaze_ts_for_alignment[
            neon_offset - self.qualisys_rec.neon_trim_begin
        ]

        self.neon_axes.plot(
            self.qualisys_rec.xdf_gaze_ts_for_alignment,
            self.qualisys_rec.xdf_neon_data_for_alignment,
        )

        self.neon_axes.plot(
            self.qualisys_rec.aligned_gaze_ts,
            self.qualisys_rec.neon_reference_gaze,
        )

        # mid_idx = len(self.qualisys_rec.aligned_gaze_ts) // 2
        # mid_ts = self.qualisys_rec.aligned_gaze_ts[mid_idx]
        # self.neon_axes.set_xlim(mid_ts - 10, mid_ts + 10)
        self.neon_axes.set_ylim(
            np.min(self.qualisys_rec.neon_reference_gaze),
            np.max(self.qualisys_rec.neon_reference_gaze),
        )

        self.neon_canvas.draw()

    def prepare_export_column(self):
        self.export_button = QPushButton("Export Synced Data")
        self.exported_label = QLabel("")

        self.export_button.clicked.connect(self.export_data)

        self.button_layout = QVBoxLayout()
        self.button_layout.addWidget(self.export_button)
        self.button_layout.addWidget(self.exported_label)

        self.main_layout.addLayout(self.button_layout)

    @Slot()
    def export_data(self):
        self.exported_label.setText("Exporting data...")

        self.export_mocap_neon_synced_csv()
        self.export_gaze_time_in_mocap_time()
        self.export_imu_in_xdf_time()

        self.exported_label.setText("Data export finished!")

        neon_rec_path_file = self.mat_export_dir / (
            Path(self.qualisys_rec.mat_path).stem + "_neon_rec_path.txt"
        )
        neon_rec_path_file.write_text(self.qualisys_rec.neon_rec_path)

    def export_mocap_neon_synced_csv(self):
        marker_df = pd.DataFrame()
        marker_df["timestamp [ns]"] = self.qualisys_rec.neon_rec.gaze.time
        for marker_name in self.qualisys_rec.marker_names:
            marker = str(marker_name[0])
            index = self.qualisys_rec.marker_indices[marker]

            marker_pos = self.qualisys_rec.qtm_marker_positions[index].squeeze()

            if marker_pos.shape[1] == 4:
                marker_pos = marker_pos.T

            # re-interpolate qtm data to correspond exactly to neon data

            fx = interp1d(
                self.qualisys_rec.aligned_qtm_ts,
                marker_pos[0, :],
                bounds_error=False,
                fill_value=np.nan,
            )
            fy = interp1d(
                self.qualisys_rec.aligned_qtm_ts,
                marker_pos[1, :],
                bounds_error=False,
                fill_value=np.nan,
            )
            fz = interp1d(
                self.qualisys_rec.aligned_qtm_ts,
                marker_pos[2, :],
                bounds_error=False,
                fill_value=np.nan,
            )

            marker_pos_x_in_neon = fx(self.qualisys_rec.aligned_gaze_ts)
            marker_pos_y_in_neon = fy(self.qualisys_rec.aligned_gaze_ts)
            marker_pos_z_in_neon = fz(self.qualisys_rec.aligned_gaze_ts)

            marker_df[f"{marker}_X"] = list(marker_pos_x_in_neon)
            marker_df[f"{marker}_Y"] = list(marker_pos_y_in_neon)
            marker_df[f"{marker}_Z"] = list(marker_pos_z_in_neon)

        marker_df.to_csv(
            self.mat_export_dir
            / (
                str(Path(self.qualisys_rec.mat_path).stem)
                + "_marker_positions_neon_ts.csv"
            )
        )

    def export_gaze_time_in_mocap_time(self):
        # map neon gaze timestamps to mocap time axis for later processing

        # find the subset of gaze timestamps in LSL timeline that overlap with the QTM
        # data

        first_overlapping_gaze_idx = 0
        if self.qualisys_rec.aligned_qtm_ts[0] >= self.qualisys_rec.aligned_gaze_ts[0]:
            for idx, elem in enumerate(self.qualisys_rec.aligned_gaze_ts):
                if self.qualisys_rec.aligned_qtm_ts[0] <= elem:
                    first_overlapping_gaze_idx = idx
                    break

        last_overlapping_gaze_idx = len(self.qualisys_rec.aligned_gaze_ts)
        if (
            self.qualisys_rec.aligned_qtm_ts[-1]
            <= self.qualisys_rec.aligned_gaze_ts[-1]
        ):
            for idx, elem in enumerate(reversed(self.qualisys_rec.aligned_gaze_ts)):
                if self.qualisys_rec.aligned_qtm_ts[-1] >= elem:
                    last_overlapping_gaze_idx = (
                        len(self.qualisys_rec.aligned_gaze_ts) - idx - 1
                    )
                    break

        gaze_time_in_qtm_xdf = self.qualisys_rec.aligned_gaze_ts[
            first_overlapping_gaze_idx:last_overlapping_gaze_idx
        ]
        gaze_time_in_qtm = gaze_time_in_qtm_xdf - self.qualisys_rec.aligned_qtm_ts[0]

        fig, ax = plt.subplots()
        ax.plot(
            self.qualisys_rec.aligned_qtm_ts,
            self.qualisys_rec.qtm_reference_positions[0, :],
            label="QTM Aligned Data",
        )
        ax.plot(
            self.qualisys_rec.aligned_gaze_ts,
            self.qualisys_rec.neon_reference_gaze,
            label="Gaze Aligned Datr",
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Data [a.u.]")
        plt.title("Synced Gaze and MoCap Data")
        plt.legend()
        plt.show()

        gaze_timing_data = {
            "first_overlapping_gaze_idx": first_overlapping_gaze_idx,
            "last_overlapping_gaze_idx": last_overlapping_gaze_idx,
            "first_overlapping_gaze_xdf_timestamp": self.qualisys_rec.aligned_gaze_ts[
                first_overlapping_gaze_idx
            ],
            "last_overlapping_gaze_xdf_timestamp": self.qualisys_rec.aligned_gaze_ts[
                last_overlapping_gaze_idx - 1
            ],
            "overlapping_gaze_time_in_xdf": gaze_time_in_qtm_xdf,
            "overlapping_gaze_time_in_qualisys": gaze_time_in_qtm,
            "first_qtm_xdf_timestamp": self.qualisys_rec.aligned_qtm_ts[0],
            "gaze_time_in_xdf": self.qualisys_rec.aligned_gaze_ts,
        }

        pkl_path = Path(self.xdf_export_dir) / (
            Path(self.qualisys_rec.mat_path).stem + "_gaze_timing_data.pkl"
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(gaze_timing_data, f)

    def export_imu_in_xdf_time(self):
        f_imu_time = interp1d(
            self.qualisys_rec.neon_rec.gaze.time,
            self.qualisys_rec.aligned_gaze_ts,
            bounds_error=False,
            fill_value=np.nan,
        )

        imu_time_in_xdf = f_imu_time(self.qualisys_rec.neon_rec.imu.time)
        np.save(
            self.xdf_export_dir
            / (str(Path(args["mocap_mat_path"]).stem) + "_imu_time_in_xdf.npy"),
            imu_time_in_xdf,
        )


if __name__ == "__main__":
    # parse args

    parser = argparse.ArgumentParser(
        description="Converts Qualisys MoCap data to a time-synced CSV file."
    )

    parser.add_argument(
        "-m",
        "--mocap_mat_path",
        help="The path to the Qualisys data (MAT file)",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--neon_rec_path",
        help="The path to the associated Neon Native Recording Data",
        required=True,
    )
    parser.add_argument(
        "-x",
        "--xdf_path",
        required=True,
        help="The XDF file produced by LabRecorder of Lab Streaming Layer",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        help="A config file containing the parameters that remain constant between \
    sessions.",
        required=True,
    )

    args = vars(parser.parse_args())

    mocap_mat_path = args["mocap_mat_path"]
    xdf_path = args["xdf_path"]
    neon_rec_path = args["neon_rec_path"]
    config_path = args["config_path"]

    qualisys_rec = data_loader.QualisysRecording(
        xdf_path,
        None,
        mocap_mat_path,
        neon_rec_path,
        config_path,
    )

    mat_export_dir = Path(mocap_mat_path).parent
    xdf_export_dir = Path(xdf_path).parent

    app = QApplication(sys.argv)
    window = MainWindow(qualisys_rec, mat_export_dir, xdf_export_dir)
    window.show()
    sys.exit(app.exec())
