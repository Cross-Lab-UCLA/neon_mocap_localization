import numpy as np


class MocapIRMarker:
    """
    Holds the timeseries of positions of an IR Marker
    """

    def __init__(self, Xs, Ys, Zs, id):
        self.Xs = Xs
        self.Ys = Ys
        self.Zs = Zs

        self.position = np.array([Xs, Ys, Zs])

        self.id = id


class MocapHead:
    def __init__(self):
        self.markers = []

    def add_marker(self, marker):
        self.markers.append(marker)


class MocapAprilTag:
    def __init__(self, tag_id):
        self.markers = []
        self.center = np.array([0, 0, 0])
        self.tag_id = tag_id

    def add_marker(self, marker):
        self.markers.append(marker)

    def estimate_tag_center(self):
        pos = np.array(
            [
                [marker.Xs for marker in self.markers],
                [marker.Ys for marker in self.markers],
                [marker.Zs for marker in self.markers],
            ]
        )
        self.center = np.mean(pos, axis=1)

    def estimate_size_mm(self):
        """
        Estimate tag size [m] from mocap data.
        """

        tag_marker1_pos = np.array(
            [
                self.markers[0].Xs,
                self.markers[0].Ys,
                self.markers[0].Zs,
            ]
        )
        tag_marker2_pos = np.array(
            [
                self.markers[1].Xs,
                self.markers[1].Ys,
                self.markers[1].Zs,
            ]
        )

        self.tag_size = np.sqrt(
            np.sum(
                np.power(
                    tag_marker1_pos - tag_marker2_pos,
                    2,
                )
            )
        )


class MocapSurface:
    def __init__(self):
        self.apriltags = []

    def add_apriltag(self, apriltag):
        self.apriltags.append(apriltag)
