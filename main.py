import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import urllib.request
import os

from connections import HAND_CONNECTIONS, FINGERTIP_IDS

# Download model if not present
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1 # currently just using 1
)

cap = cv2.VideoCapture(0)
pTime = 0

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        for hand_landmarks in result.hand_landmarks:
            # Convert to pixel coords
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            # Draw connections
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, pts[a], pts[b], (0, 255, 255), 2)

            # Draw all landmarks
            for i, (cx, cy) in enumerate(pts):
                if i in FINGERTIP_IDS:
                    cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
                else:
                    cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime else 0
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# cleanup
cap.release()
cv2.destroyAllWindows()
