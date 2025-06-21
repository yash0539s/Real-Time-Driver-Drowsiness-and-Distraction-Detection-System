import cv2, torch, numpy as np
import mediapipe as mp

from utils.logger import init_logger
from utils.helpers import load_config, preprocess_image
from models.driver_model import DriverMonitorModel
from src.face_landmark import init_face_mesh, extract_landmarks
from src.drowsiness_detector import AdvancedDrowsinessDetector
from src.head_pose_estimator import estimate_head_pose
from src.distraction_detector import AdvancedDistractionDetector
from src.face_id_verification import load_known_faces, recognize_faces, enroll_new_driver
from src.alert_system import AlertSystem

def main():
    cfg = load_config()
    log = init_logger()
    log.info("🚀 Driver Monitoring Starting...")

    # Load CNN model
    model = DriverMonitorModel(num_classes=3)
    ckpt = torch.load("models/epoch_24_ckpt.pth.tar", map_location="cpu")
    fixed_state_dict = {
        k.replace("gaze_network.", "backbone."): v
        for k, v in ckpt["model_state"].items()
        if not (k.endswith("fc.weight") or k.endswith("fc.bias"))
    }
    model.load_state_dict(fixed_state_dict, strict=False)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Init components
    fm = init_face_mesh()
    dod = AdvancedDrowsinessDetector(cfg['detection']['ear_threshold'], cfg['detection']['consecutive_frames_threshold'])
    dd = AdvancedDistractionDetector()
    al = AlertSystem(cfg['alert']['sound_file'], cfg['alert']['buzzer_enabled'])

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    known_encs, names = load_known_faces() if cfg['face_id']['enable'] else ([], [])

    cap = cv2.VideoCapture(cfg['camera']['id'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg['camera']['height'])
    cap.set(cv2.CAP_PROP_FPS, cfg['camera']['fps'])

    if not cap.isOpened():
        log.error("❌ Camera can't be opened")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        status = "No Face Detected"
        lm = extract_landmarks(frame, fm)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb)
        hand_landmarks = hand_results.multi_hand_landmarks[0].landmark if hand_results.multi_hand_landmarks else None

        if lm:
            drowsy = dod.update(lm)
            rmat, _ = estimate_head_pose(lm, frame.shape)
            distracted = dd.is_distracted(rmat, lm, hand_landmarks, image_shape=frame.shape[:2], debug=True)


            inp = preprocess_image(frame)
            pred = torch.argmax(model(inp.to(device)), dim=1).item()
            cls = ["Alert", "Drowsy", "Distracted"][pred]

            if drowsy or cls == "Drowsy":
                status = "🚨 Drowsy"
                al.trigger()
            elif distracted or cls == "Distracted":
                status = "🚨 Using Phone" if hand_landmarks else "🚨 Distracted"
                al.trigger()
            else:
                status = "✅ Alert"

            # Face ID Recognition
            if names:
                results = recognize_faces(frame, known_encs, names)
                for res in results:
                    name = res['name']
                    top, right, bottom, left = res['box']
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    status = f"{name}: {status}"

        # Overlay UI
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if "🚨" in status else (0, 255, 0), 2)
        cv2.putText(frame, "Press 'n' to enroll new driver", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

        cv2.imshow("DriverMonitor", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('n'):
            new_name = input("Enter new driver name: ")
            enroll_new_driver(cap, new_name)
            known_encs, names = load_known_faces()  # Reload updated faces

    cap.release()
    cv2.destroyAllWindows()
    fm.close()
    hands.close()

if __name__ == '__main__':
    main()
