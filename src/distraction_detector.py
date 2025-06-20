import numpy as np

class DistractionDetector:
    def __init__(self, yaw_thresh=0.35, phone_dist_thresh=0.1):
        self.yaw_thresh = yaw_thresh
        self.phone_dist_thresh = phone_dist_thresh

    def is_distracted(self, rotation_mat, face_landmarks, hand_landmarks):
        yaw = -np.arcsin(rotation_mat[2][0])
        looking_away = abs(yaw) > self.yaw_thresh

        # Phone detection by checking distance from hand to face center
        phone_detected = False
        if hand_landmarks:
            face_x = np.mean([p[0] for p in face_landmarks])
            face_y = np.mean([p[1] for p in face_landmarks])
            hx = hand_landmarks[8].x  # Index finger tip
            hy = hand_landmarks[8].y
            dist = np.sqrt((hx - face_x)**2 + (hy - face_y)**2)
            phone_detected = dist < self.phone_dist_thresh

        return looking_away or phone_detected
