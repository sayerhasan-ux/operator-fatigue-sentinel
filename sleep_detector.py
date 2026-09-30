import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Load Face Landmarker
MODEL_FILE = "face_landmarker.task"
base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# 2. Eye Aspect Ratio (EAR) Calculation
def get_ear(landmarks, top_idx, bot_idx, left_idx, right_idx, w, h):
    top = (int(landmarks[top_idx].x * w), int(landmarks[top_idx].y * h))
    bot = (int(landmarks[bot_idx].x * w), int(landmarks[bot_idx].y * h))
    left = (int(landmarks[left_idx].x * w), int(landmarks[left_idx].y * h))
    right = (int(landmarks[right_idx].x * w), int(landmarks[right_idx].y * h))

    vertical = math.dist(top, bot)
    horizontal = math.dist(left, right)
    return 0.0 if horizontal == 0 else vertical / horizontal

# Calibration
EAR_THRESHOLD = 0.25
CLOSED_LIMIT = 20
closed_frames = 0

cap = cv2.VideoCapture(0)
print("Sleep Sentinel running. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    if result.face_landmarks:
        face = result.face_landmarks[0]
        left_ear = get_ear(face, 159, 145, 33, 133, w, h)
        right_ear = get_ear(face, 386, 374, 362, 263, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            closed_frames += 1
        else:
            closed_frames = max(0, closed_frames - 2)

        if closed_frames >= CLOSED_LIMIT:
            cv2.putText(frame, "STATUS: SLEEP DETECTED!", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
        else:
            cv2.putText(frame, f"STATUS: AWAKE ({closed_frames})", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "STATUS: NO FACE DETECTED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    cv2.imshow("CCTV Workplace Safety - Sleep Sentinel", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

