import numpy as np
from scipy.spatial import distance as dist

class AdvancedDrowsinessDetector:
    def __init__(self, ear_thresh=0.25, mar_thresh=0.7, blink_consec_frames=3, yawn_consec_frames=15):
        self.ear_thresh = ear_thresh
        self.mar_thresh = mar_thresh
        self.blink_consec_frames = blink_consec_frames
        self.yawn_consec_frames = yawn_consec_frames
        self.blink_counter = 0
        self.yawn_counter = 0

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[13], mouth[19])
        B = dist.euclidean(mouth[14], mouth[18])
        C = dist.euclidean(mouth[15], mouth[17])
        D = dist.euclidean(mouth[12], mouth[16])
        return (A + B + C) / (3.0 * D)

    def update(self, landmarks, debug=False):
        left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
        right_eye = [landmarks[i] for i in [362, 385, 387, 263, 373, 380]]
        mouth = [landmarks[i] for i in range(48, 68)]  # Change if not using 68-point model

        ear = (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)) / 2.0
        mar = self.mouth_aspect_ratio(mouth)

        drowsy = False

        if ear < self.ear_thresh:
            self.blink_counter += 1
        else:
            self.blink_counter = 0

        if mar > self.mar_thresh:
            self.yawn_counter += 1
        else:
            self.yawn_counter = 0

        if self.blink_counter >= self.blink_consec_frames or self.yawn_counter >= self.yawn_consec_frames:
            drowsy = True

        if debug:
            print(f"[DEBUG] EAR: {ear:.2f}, MAR: {mar:.2f}, BlinkCount: {self.blink_counter}, YawnCount: {self.yawn_counter}")

        return drowsy
