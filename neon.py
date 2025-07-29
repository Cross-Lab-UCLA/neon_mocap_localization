class Neon:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.pose_in_tags = []
        self.pose_in_surface = None
        self.pose_in_optitrack = None

    def add_pose_in_tag(self, pose):
        self.pose_in_tags.append(pose)

    def set_pose_in_surface(self, pose):
        self.pose_in_surface = pose

    def calculate_pose_in_optitrack(self, display_pose_optitrack):
        if self.pose_in_surface is None:
            raise ValueError("Pose in surface coordinates is not set.")

        # convert neon pose in display surface coordinates to optitrack format
        neon_pose_in_display_surface_optitrack = (
            self.pose_in_surface.to_pupil_labs_optitrack_format()
        )
        self.pose_in_optitrack = display_pose_optitrack.apply(
            neon_pose_in_display_surface_optitrack
        )
