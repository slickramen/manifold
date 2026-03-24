import argparse
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import urllib.request
import os
import math

from pynput.mouse import Controller, Button
import Quartz

from connections import HAND_CONNECTIONS, FINGERTIP_IDS

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='Show camera feed window')
args = parser.parse_args()

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
    num_hands=1
)

main_display = Quartz.CGMainDisplayID()
sw = Quartz.CGDisplayPixelsWide(main_display)
sh = Quartz.CGDisplayPixelsHigh(main_display)

mouse = Controller()
smooth_x, smooth_y = sw // 2, sh // 2
SMOOTHING = 0.35

ZONE_LEFT, ZONE_RIGHT = 0.1, 0.9
ZONE_TOP, ZONE_BOTTOM = 0.1, 0.9

GESTURE_BUFFER = 5
gesture_history = []
last_gesture = None


def fingers_extended(hand_landmarks):
    tips     = [4, 8, 12, 16, 20]
    knuckles = [3, 6, 10, 14, 18]

    extended = []
    for i, (tip, knuckle) in enumerate(zip(tips, knuckles)):
        if i == 0:
            extended.append(hand_landmarks[tip].x < hand_landmarks[knuckle].x)
        else:
            extended.append(hand_landmarks[tip].y < hand_landmarks[knuckle].y)

    return extended


def pinch_distance(hand_landmarks, tip_a, tip_b):
    a = hand_landmarks[tip_a]
    b = hand_landmarks[tip_b]
    return math.hypot(a.x - b.x, a.y - b.y)


def detect_gesture(hand_landmarks, pinch_threshold=0.05):
    if pinch_distance(hand_landmarks, 4, 8) < pinch_threshold:
        return "PINCH_INDEX"
    if pinch_distance(hand_landmarks, 4, 12) < pinch_threshold:
        return "PINCH_MIDDLE"

    extended = fingers_extended(hand_landmarks)
    fingers = extended[1:]  # ignore thumb

    if not any(fingers):
        return "FIST"
    if all(fingers):
        return "OPEN"
    # if fingers == [True, False, False, False]:
    #     return "POINTING"
    # if fingers == [False, True, False, False]:
    #     return "NAUGHTY"
    # if fingers == [True, True, False, False]:
    #     return "PEACE"
    # if fingers == [True, False, False, True]:
    #     return "ROCK"

    return "NONE"


def stable_gesture(current):
    gesture_history.append(current)
    if len(gesture_history) > GESTURE_BUFFER:
        gesture_history.pop(0)
    if gesture_history.count(gesture_history[-1]) == GESTURE_BUFFER:
        return gesture_history[-1]
    return None


def handle_gesture(gesture, last_gesture):
    if gesture == last_gesture:
        return

    # Release drag if fist is released
    if last_gesture == "FIST":
        mouse.release(Button.left)

    if gesture == "PINCH_INDEX":
        mouse.click(Button.left)
    elif gesture == "PINCH_MIDDLE":
        mouse.click(Button.right)
    elif gesture == "FIST":
        mouse.press(Button.left)


cap = cv2.VideoCapture(0)
pTime = 0

PALM_IDS = [0, 5, 9, 13, 17]

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
            gesture = stable_gesture(detect_gesture(hand_landmarks))

            handle_gesture(gesture, last_gesture)
            last_gesture = gesture

            # Cursor control
            anchor_x = sum(hand_landmarks[i].x for i in PALM_IDS) / len(PALM_IDS)
            anchor_y = sum(hand_landmarks[i].y for i in PALM_IDS) / len(PALM_IDS)

            raw_x = (anchor_x - ZONE_LEFT) / (ZONE_RIGHT - ZONE_LEFT)
            raw_y = (anchor_y - ZONE_TOP) / (ZONE_BOTTOM - ZONE_TOP)
            raw_x = max(0.0, min(1.0, raw_x))
            raw_y = max(0.0, min(1.0, raw_y))

            target_x = int(raw_x * sw)
            target_y = int(raw_y * sh)
            smooth_x += (target_x - smooth_x) * SMOOTHING
            smooth_y += (target_y - smooth_y) * SMOOTHING

            mouse.position = (int(smooth_x), int(smooth_y))

            if args.debug:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(img, pts[a], pts[b], (0, 255, 255), 2)
                for i, (cx, cy) in enumerate(pts):
                    if i in FINGERTIP_IDS:
                        cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
                    else:
                        cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)
                if gesture:
                    cv2.putText(img, gesture, (10, 110),
                                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        if not result.hand_landmarks and last_gesture == "FIST":
            mouse.release(Button.left)
            last_gesture = None

        if args.debug:
            zone_x1 = int(ZONE_LEFT * w)
            zone_y1 = int(ZONE_TOP * h)
            zone_x2 = int(ZONE_RIGHT * w)
            zone_y2 = int(ZONE_BOTTOM * h)
            cv2.rectangle(img, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 0), 2)

            cTime = time.time()
            fps = 1 / (cTime - pTime) if pTime else 0
            pTime = cTime

            cv2.putText(img, str(int(fps)), (10, 70),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            cv2.imshow("Debug", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
if args.debug:
    cv2.destroyAllWindows()
