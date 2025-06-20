import numpy as np
class DistractionDetector:
    def __init__(self, yaw_thresh=0.35):
        self.yaw_thresh = yaw_thresh

    def is_looking_away(self, rotation_mat):
        yaw = -np.arcsin(rotation_mat[2][0])
        return abs(yaw) > self.yaw_thresh
