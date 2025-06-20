import cv2
import numpy as np

# 3D model and 2D pts for head pose
_model_points = np.array([
    (0.0, 0.0, 0.0),     # nose tip
    (0.0, -330.0, -65.0),# chin
    (-225.0, 170.0, -135.0), # left eye
    (225.0, 170.0, -135.0),  # right eye
    (-150.0, -150.0, -125.0),# left mouth
    (150.0, -150.0, -125.0)  # right mouth
], dtype=np.float64)

def estimate_head_pose(landmarks, size):
    image_pts = np.array([
        landmarks[1], landmarks[152],
        landmarks[33], landmarks[263],
        landmarks[78], landmarks[308]
    ], dtype="double")

    focal = size[1]
    center = (size[1] / 2, size[0] / 2)
    cam_mtx = np.array([[focal, 0, center[0]],
                        [0, focal, center[1]],
                        [0, 0, 1]], dtype="double")
    _, rot, trans = cv2.solvePnP(_model_points, image_pts, cam_mtx, None)
    rot_mat, _ = cv2.Rodrigues(rot)
    return rot_mat, trans
