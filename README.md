#  Real-Time Driver Monitoring System

A comprehensive real-time driver monitoring system that detects **drowsiness**, **distraction**, **phone usage**, and verifies driver identity using **face recognition**. Built with **MediaPipe**, **OpenCV**, **PyTorch**, and **deep learning**, the system ensures safety through intelligent multi-modal monitoring.

---

##  Features

-  **Drowsiness Detection** using Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) with blink and yawn counting.
-  **Distraction Detection** using head pose estimation (yaw, pitch) and CNN classification.
-  **Phone Usage Detection** based on hand-to-face proximity using MediaPipe Hands.
-  **Deep CNN Prediction** via ResNet50 for classification into: `Alert`, `Drowsy`, or `Distracted`.
-  **Face ID Verification**: Recognize known drivers, and allow real-time enrollment with `'n'` key.
-  **Alert System**: Sound buzzer or play alert audio if driver is drowsy, distracted, or using phone.
-  **Live Webcam Interface** with clear UI overlays for real-time feedback and debugging.

---

## 📁 Project Structure

```
driver_monitoring/
│
├── models/
│   └── driver_model.py         # ResNet-50 classifier
│   └── epoch_24_ckpt.pth.tar   # Pretrained model checkpoint
│
├── src/
│   ├── drowsiness_detector.py     # EAR/MAR-based drowsiness detection
│   ├── distraction_detector.py    # Head pose + phone detection
│   ├── face_id_verification.py    # Face enrollment & recognition
│   ├── head_pose_estimator.py     # Estimate yaw/pitch/roll
│   ├── alert_system.py            # Sound alert or buzzer
│   └── face_landmark.py           # MediaPipe landmark utilities
│
├── utils/
│   ├── logger.py               # Logger initialization
│   ├── helpers.py              # Config loader & preprocessing
│   └── config.yaml             # All thresholds & parameters
│
├── main.py                    # 🧠 Core script: runs the system
├── requirements.txt           # Dependencies
└── README.md                  # 📘 You are here
```

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

**Main Libraries:**

- `torch`, `torchvision`
- `opencv-python`
- `mediapipe`
- `numpy`, `scipy`
- `face_recognition`
- `pygame` or `playsound` (for alerts)

---

## 🧠 How It Works

1. **Face & Landmark Detection** using MediaPipe Face Mesh.
2. **EAR/MAR Calculation** to detect eye closure or yawning.
3. **Head Pose Estimation** using 3D PnP (yaw, pitch, roll).
4. **Phone Usage Detection** via distance between hand landmarks and face.
5. **CNN Classification** from cropped face image into 3 classes.
6. **Face Recognition** compares real-time encoding with known drivers.
7. **Priority Fusion Logic** displays the most critical status (e.g.,  Phone >  Drowsy >  Distracted >  Alert).

---

## 🚀 Usage

```bash
python main.py
```

- Press `'q'` to quit.
- Press `'n'` to enroll a new driver face in real-time (you will be prompted to enter a name).

---

##  Output Overlay

Real-time webcam feed shows:

- Driver's status (e.g., Alert,  Drowsy,  Using Phone).
- Detected phone usage frames.
- Name of recognized driver.
- Optional debug info for EAR, MAR, yaw, pitch.

---

##  Configuration

Edit `utils/config.yaml` to change thresholds and settings:

```yaml
detection:
  ear_threshold: 0.25
  mar_threshold: 0.7
  consecutive_frames_threshold: 3
  yawn_frames_threshold: 15

camera:
  id: 0
  width: 640
  height: 480
  fps: 30

alert:
  sound_file: "assets/alert.wav"
  buzzer_enabled: true

face_id:
  enable: true
```

---

## 🧑 Model Info

- **Architecture**: ResNet-50
- **Classes**: `Alert`, `Drowsy`, `Distracted`
- **Training**: Fine-tuned on a custom dataset of driver states (expandable)

---

##  To-Do / Future Work

- Add **emotion detection** module.
- Improve **lip sync** detection for phone calls.
- Add **night mode** enhancements.
- Cloud-based **real-time monitoring dashboard**.

---

##  Final Notes

> This system is designed for **driver safety**, **fleet monitoring**, and **automotive research**. Ensure proper lighting and camera angle for best accuracy.

---

## 👨 Author

Built with by [Yash Malviya]  
