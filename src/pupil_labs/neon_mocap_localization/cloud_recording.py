import json
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd


class Scene:
    def __init__(self, rec_path: Path):
        self.rec_path = rec_path

        self.time_csv = pd.read_csv(self.rec_path / "world_timestamps.csv")

        self.idx = self.time_csv.index.to_numpy()
        self.time: npt.NDArray[np.int64] = (
            self.time_csv["timestamp [ns]"].to_numpy().astype(np.int64)
        )

        self.nframes = len(self.time_csv)

        scene_mp4_path: Path = next(iter(self.rec_path.glob("*.mp4")))

        self.container = av.open(scene_mp4_path)
        self.video_stream = self.container.streams.video[0]

        self.duration = self.video_stream.duration

    def _seek(self, vid_time: float) -> None:
        self.container.seek(
            int(vid_time / self.video_stream.time_base),  # type: ignore
            stream=self.video_stream,
        )

    def _get_img(self, vid_time: float) -> Any | None:
        exact_frame = None
        for frame in self.container.decode(self.video_stream):
            # find the first frame >= the requested timestamp
            if frame.time >= vid_time:
                exact_frame = frame
                break

        if exact_frame is not None:
            return exact_frame.to_ndarray(format="rgb24")
        else:
            return None

    def bgr_at_time(self, time: np.signedinteger) -> npt.NDArray[np.uint8] | None:
        vid_time = (time - self.time[0]) * 1e-9
        self._seek(vid_time)
        img = self._get_img(vid_time)
        if img is None:
            return None
        else:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


class Gaze:
    def __init__(self, rec_path: Path):
        self.rec_path = rec_path

        self.data_csv = pd.read_csv(rec_path / "gaze.csv")
        self.time = self.data_csv["timestamp [ns]"].to_numpy()

        self.nframes = len(self.data_csv)

        self.data = {
            "point_x": self.data_csv["gaze x [px]"],
            "point_y": self.data_csv["gaze y [px]"],
        }

    def sample(self, ts: npt.ArrayLike) -> dict[str, Any]:
        idxs = np.searchsorted(ts, self.time)

        return {
            "point_x": self.data["point_x"][idxs],
            "point_y": self.data["point_y"][idxs],
        }


class Calibration:
    def __init__(self, rec_path: Path):
        self.calib_path = rec_path / "scene_camera.json"

        scene_calib = []
        with open(self.calib_path) as f:
            scene_calib = json.load(f)

        self.scene_camera_matrix = np.array(scene_calib["camera_matrix"])
        self.scene_distortion_coefficients = np.array(
            scene_calib["distortion_coefficients"]
        )


class Events:
    def __init__(self, rec_path: Path):
        self.events_path = rec_path / "events.csv"

        self.data = pd.read_csv(self.events_path)


class CloudRecording:
    def __init__(self, directory: str):
        self.directory = Path(directory)

        self.info = []
        with open(self.directory / "info.json") as f:
            self.info = json.load(f)

        self.scene = Scene(self.directory)
        self.calibration = Calibration(self.directory)
        self.gaze = Gaze(self.directory)
        self.events = Events(self.directory)
