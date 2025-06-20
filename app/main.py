import cv2, torch, numpy as np
from utils.logger import init_logger
from utils.helpers import load_config, preprocess_image
from models.driver_model import DriverMonitorModel
from src.face_landmark import init_face_mesh, extract_landmarks
from src.drowsiness_detector import DrowsinessDetector
from src.head_pose_estimator import estimate_head_pose
from src.distraction_detector import DistractionDetector
from src.face_id_verification import load_known_faces, recognize_faces
from src.alert_system import AlertSystem

def main():
    cfg = load_config()
    log = init_logger()
    log.info("Driver Monitoring Starting...")

    model = DriverMonitorModel(num_classes=3)
    ckpt = torch.load('models/epoch_24_ckpt.pth.tar', map_location='cpu')

    fixed_state_dict = {
        k.replace("gaze_network.", "backbone."): v
        for k, v in ckpt["model_state"].items()
    }

    model.load_state_dict(fixed_state_dict, strict=False)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    fm = init_face_mesh()
    dod = DrowsinessDetector(cfg['detection']['ear_threshold'], cfg['detection']['consecutive_frames_threshold'])
    dd = DistractionDetector()
    al = AlertSystem(cfg['alert']['sound_file'], cfg['alert']['buzzer_enabled'])

    if cfg['face_id']['enable']:
        known_encs, names = load_known_faces()
    else:
        known_encs, names = [], []

    cap = cv2.VideoCapture(cfg['camera']['id'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg['camera']['height'])
    cap.set(cv2.CAP_PROP_FPS, cfg['camera']['fps'])

    if not cap.isOpened():
        log.error("❌ Camera can't be opened")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break

        lm = extract_landmarks(frame, fm)
        status = "No Face"
        if lm:
            drowsy = dod.update(lm)
            rmat, _ = estimate_head_pose(lm, frame.shape)
            distracted = dd.is_looking_away(rmat)

            inp = preprocess_image(frame)
            pred = torch.argmax(model(inp.to(device)), dim=1).item()
            cls = ["Alert","Drowsy","Distracted"][pred]

            if drowsy or cls=="Drowsy":
                status = "Drowsy"
                al.trigger()
            elif distracted or cls=="Distracted":
                status = "Distracted"
                al.trigger()
            else:
                status = "Alert"

            if names:
                pid = recognize_faces(frame, known_encs, names)
                status = f"{pid}: {status}"

        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("DriverMonitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()
    fm.close()

if __name__ == '__main__':
    main()
