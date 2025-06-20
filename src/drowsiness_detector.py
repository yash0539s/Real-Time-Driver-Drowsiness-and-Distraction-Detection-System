import numpy as np

def eye_aspect_ratio(eye):
    A, B = np.linalg.norm(eye[1]-eye[5]), np.linalg.norm(eye[2]-eye[4])
    C = np.linalg.norm(eye[0]-eye[3])
    return (A + B) / (2.0 * C)

class DrowsinessDetector:
    def __init__(self, ear_thresh=0.21, consec_frames=15):
        self.ear_thresh, self.consec_frames = ear_thresh, consec_frames
        self.counter = 0

    def update(self, landmarks):
        # landmarks: full list, eye points indices manually defined
        left = np.array([landmarks[i] for i in [33, 160, 158, 133, 153, 144]])
        right = np.array([landmarks[i] for i in [362, 385, 387, 263, 373, 380]])
        l_ear, r_ear = eye_aspect_ratio(left), eye_aspect_ratio(right)
        ear = (l_ear + r_ear) / 2.0

        if ear < self.ear_thresh:
            self.counter += 1
        else:
            self.counter = 0

        return self.counter >= self.consec_frames
