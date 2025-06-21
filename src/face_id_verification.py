import face_recognition
import os
import numpy as np
import cv2
from datetime import datetime

def load_known_faces(path="data/known_faces"):
    encodings, names = [], []
    for file in os.listdir(path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(path, file)
            image = face_recognition.load_image_file(img_path)
            enc = face_recognition.face_encodings(image)
            if enc:
                encodings.append(enc[0])
                names.append(os.path.splitext(file)[0])
    return encodings, names

def recognize_faces(frame, known_encodings, known_names, tolerance=0.45):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for face_encoding, face_location in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_idx = np.argmin(distances)

        if distances[best_match_idx] < tolerance:
            name = known_names[best_match_idx]
            confidence = 1.0 - distances[best_match_idx]
        else:
            name = "Unknown"
            confidence = 0.0

        results.append({
            "name": name,
            "confidence": round(confidence, 2),
            "box": face_location
        })

    return results

def enroll_new_driver(cap, name, save_path="data/known_faces"):
    print(f"[INFO] Capturing image for new driver: {name}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera failed.")
            return
        cv2.imshow("Capture - Press 's' to save", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            img_path = os.path.join(save_path, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"[✓] Saved new driver image to {img_path}")
            break
        elif key & 0xFF == ord('q'):
            print("[INFO] Cancelled enrollment.")
            break
    cv2.destroyWindow("Capture - Press 's' to save")
