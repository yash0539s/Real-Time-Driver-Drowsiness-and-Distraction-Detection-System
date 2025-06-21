import numpy as np

class AdvancedDistractionDetector:
    def __init__(self, yaw_thresh=0.35, pitch_thresh=0.25, phone_dist_thresh=0.12):
        self.yaw_thresh = yaw_thresh
        self.pitch_thresh = pitch_thresh
        self.phone_dist_thresh = phone_dist_thresh

    def is_distracted(self, rotation_mat, face_landmarks, hand_landmarks, image_shape=(480, 640), debug=False):
        image_h, image_w = image_shape

        # Calculate yaw, pitch, roll
        yaw = -np.arcsin(rotation_mat[2][0])
        pitch = np.arctan2(-rotation_mat[2][1], rotation_mat[2][2])
        roll = np.arctan2(-rotation_mat[1][0], rotation_mat[0][0])

        looking_away = abs(yaw) > self.yaw_thresh or abs(pitch) > self.pitch_thresh

        phone_detected = False
        if hand_landmarks:
            # Get normalized hand coordinates
            hand_x = np.mean([hand_landmarks[8].x, hand_landmarks[12].x])
            hand_y = np.mean([hand_landmarks[8].y, hand_landmarks[12].y])

            # Normalize face coordinates (convert pixel to [0, 1])
            key_face_indices = [1, 33, 263, 168, 152]
            face_x = np.mean([face_landmarks[i][0] / image_w for i in key_face_indices])
            face_y = np.mean([face_landmarks[i][1] / image_h for i in key_face_indices])

            # Euclidean distance in normalized space
            dist = np.sqrt((hand_x - face_x)**2 + (hand_y - face_y)**2)
            phone_detected = dist < self.phone_dist_thresh

            if debug:
                print(f"[DEBUG] Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, PhoneDist: {dist:.3f}")
                print(f"[DEBUG] LookingAway: {looking_away}, PhoneDetected: {phone_detected}")

        return looking_away or phone_detected
