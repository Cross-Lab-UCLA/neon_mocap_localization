import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd


class Scene:
    def __init__(self, rec_path):
        self.rec_path = rec_path

        self.time_csv = pd.read_csv(self.rec_path / "world_timestamps.csv")

        self.idx = self.time_csv.index.to_numpy()
        self.time = self.time_csv["timestamp [ns]"].to_numpy()

        self.nframes = len(self.time_csv)

        scene_mp4_path = next(iter(self.rec_path.glob("*.mp4")))

        self.container = av.open(scene_mp4_path)
        self.video_stream = self.container.streams.video[0]

        self.duration = self.video_stream.duration

    def _seek(self, vid_time):
        self.container.seek(
            int(vid_time / self.video_stream.time_base), stream=self.video_stream
        )

    def _get_img(self, vid_time):
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

    def bgr_at_time(self, time):
        vid_time = (time - self.time[0]) * 1e-9
        self._seek(vid_time)
        return cv2.cvtColor(self._get_img(vid_time), cv2.COLOR_RGB2BGR)


class Gaze:
    def __init__(self, rec_path):
        self.rec_path = rec_path

        self.data_csv = pd.read_csv(rec_path / "gaze.csv")
        self.time = self.data_csv["timestamp [ns]"].to_numpy()

        self.nframes = len(self.data_csv)

        self.data = {}
        self.data["point_x"] = self.data_csv["gaze x [px]"]
        self.data["point_y"] = self.data_csv["gaze y [px]"]


class Calibration:
    def __init__(self, rec_path):
        self.calib_path = rec_path / "scene_camera.json"

        scene_calib = []
        with open(self.calib_path) as f:
            scene_calib = json.load(f)

        self.scene_camera_matrix = np.array(scene_calib["camera_matrix"])
        self.scene_distortion_coefficients = np.array(
            scene_calib["distortion_coefficients"]
        )


class Events:
    def __init__(self, rec_path):
        self.events_path = rec_path / "events.csv"

        self.data = pd.read_csv(self.events_path)


class CloudRecording:
    def __init__(self, directory):
        self.directory = Path(directory)

        self.info = []
        with open(self.directory / "info.json") as f:
            self.info = json.load(f)

        self.scene = Scene(self.directory)
        self.calibration = Calibration(self.directory)
        self.gaze = Gaze(self.directory)
        self.events = Events(self.directory)
