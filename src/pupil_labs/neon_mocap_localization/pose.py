from typing import Self

import numpy as np
import numpy.typing as npt


class Pose:
    def __init__(
        self, position: npt.NDArray[np.float64], rotation: npt.NDArray[np.float64]
    ):
        self.position = position
        self.rotation = rotation

    def __repr__(self: Self) -> str:
        return f"Pose(\nposition={self.position},\n\nrotation={self.rotation}\n)"

    def inverse(self: Self) -> "Pose":
        """Return the inverse of the pose."""
        inv_rotation = self.rotation.T
        inv_position = -inv_rotation @ self.position
        return Pose(position=inv_position, rotation=inv_rotation)

    def to_matrix(self: Self) -> npt.NDArray[np.float64]:
        """Convert the pose to a transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.position
        return matrix

    def apply(self, pose: Self) -> "Pose":
        """Apply this pose to another pose."""
        new_position = self.rotation @ pose.position + self.position
        new_rotation = self.rotation @ pose.rotation

        return Pose(position=new_position, rotation=new_rotation)
