import mediapipe as mp
mpfm = mp.solutions.face_mesh

def init_face_mesh():
    return mpfm.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_landmarks(image, face_mesh):
    import cv2
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    import numpy as np
    lm = res.multi_face_landmarks[0].landmark
    h, w, _ = image.shape
    coords = [(int(p.x*w), int(p.y*h)) for p in lm]
    return coords
