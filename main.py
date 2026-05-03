import argparse
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import urllib.request
import os
import math
import subprocess

from pynput.mouse import Controller, Button
import Quartz

from connections import HAND_CONNECTIONS, FINGERTIP_IDS

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Show camera feed window")
args = parser.parse_args()

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH,
    )

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_hands=1
)

main_display = Quartz.CGMainDisplayID()
sw = Quartz.CGDisplayPixelsWide(main_display)
sh = Quartz.CGDisplayPixelsHigh(main_display)

mouse = Controller()

smooth_x, smooth_y = sw // 2, sh // 2

# Smoothing constants
SMOOTH_MIN = 0.15
SMOOTH_MAX = 0.65
VELOCITY_THRESHOLD_LOW = 5
VELOCITY_THRESHOLD_HIGH = 40

ZONE_LEFT, ZONE_RIGHT = 0.1, 0.9
ZONE_TOP, ZONE_BOTTOM = 0.1, 0.9

# Edge dead zone
EDGE_ZONE = 0.05  # 5% of screen on each side
EDGE_DAMPING = 0.3  # alpha multiplier at the very edge

GESTURE_BUFFER = 3
gesture_history = []
last_gesture = None

# Long-pinch drag state
pinch_start_time = None
LONG_PINCH_DURATION = 0.2
is_dragging = False

# Mission Control cooldown
last_mission_control_time = 0.0
MISSION_CONTROL_COOLDOWN = 1.0  # seconds

PALM_IDS = [0, 5, 9, 13, 17]


# Hand metric helpers
def palm_width(hand_landmarks):
    """3D distance from wrist (0) to middle MCP (9) — scale reference."""
    a = hand_landmarks[0]
    b = hand_landmarks[9]
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def normalised_pinch_distance(hand_landmarks, tip_a, tip_b):
    """3D pinch distance normalised by palm width."""
    a = hand_landmarks[tip_a]
    b = hand_landmarks[tip_b]
    raw = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
    pw = palm_width(hand_landmarks) or 1e-6
    return raw / pw


def hand_rotation_factor(hand_landmarks):
    """
    How face-on the hand is to the camera, in [0.0, 1.0].
    Compares x-spread of index MCP (5) to pinky MCP (17), normalised by palm width.
    1.0 = fully face-on, 0.0 = fully edge-on.
    """
    x_spread = abs(hand_landmarks[5].x - hand_landmarks[17].x)
    pw = palm_width(hand_landmarks) or 1e-6
    normalised = x_spread / pw

    return max(0.0, min(1.0, (normalised - 0.15) / 0.45))


def finger_angle(hand_landmarks, a, b, c):
    """
    Angle in degrees at landmark b between vectors b->a and b->c using 3D coords.
    180 = perfectly straight.
    """
    ax = hand_landmarks[a].x - hand_landmarks[b].x
    ay = hand_landmarks[a].y - hand_landmarks[b].y
    az = hand_landmarks[a].z - hand_landmarks[b].z
    cx = hand_landmarks[c].x - hand_landmarks[b].x
    cy = hand_landmarks[c].y - hand_landmarks[b].y
    cz = hand_landmarks[c].z - hand_landmarks[b].z
    dot = ax * cx + ay * cy + az * cz
    mag = (math.sqrt(ax**2 + ay**2 + az**2) * math.sqrt(cx**2 + cy**2 + cz**2)) or 1e-6
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def finger_straightness(hand_landmarks):
    """
    Straightness score [0.0-1.0] per finger (index, middle, ring, pinky).
    Measures 3D angle at PIP joint. Maps [90deg, 180deg] -> [0.0, 1.0].
    Resting ~0.60-0.75; deliberate splay ~0.88+; fist ~0.05-0.25.
    """
    joints = [
        (5, 6, 8),  # index
        (9, 10, 12),  # middle
        (13, 14, 16),  # ring
        (17, 18, 20),  # pinky
    ]
    scores = []
    for mcp, pip, tip in joints:
        angle = finger_angle(hand_landmarks, mcp, pip, tip)
        scores.append(max(0.0, min(1.0, (angle - 90.0) / 90.0)))
    return scores


def fingertips_above_palm(hand_landmarks):
    """
    True if all 4 fingertips are above the palm baseline.
    Guards against tight fists where curled tips wrap back to a high angle score.
    """
    palm_y = (hand_landmarks[0].y + hand_landmarks[9].y) / 2.0
    return all(hand_landmarks[tip].y < palm_y for tip in [8, 12, 16, 20])


# Gesture detection
PINCH_THRESHOLD = 0.20
PINCH_THRESHOLD_MAX = 0.35
EXTEND_THRESHOLD = 0.92


def effective_pinch_threshold(hand_landmarks):
    """
    Scale pinch threshold with hand rotation.
    Edge-on (factor~0) → tight threshold (suppresses rotation false-positives).
    Face-on (factor~1) → relaxed threshold (easy intentional pinch).
    """
    factor = hand_rotation_factor(hand_landmarks)
    return PINCH_THRESHOLD + factor * (PINCH_THRESHOLD_MAX - PINCH_THRESHOLD)


def detect_gesture(hand_landmarks):
    scores = finger_straightness(hand_landmarks)

    if all(s >= EXTEND_THRESHOLD for s in scores) and fingertips_above_palm(
        hand_landmarks
    ):
        return "OPEN"

    threshold = effective_pinch_threshold(hand_landmarks)
    if normalised_pinch_distance(hand_landmarks, 4, 8) < threshold:
        return "PINCH_INDEX"
    if normalised_pinch_distance(hand_landmarks, 4, 12) < threshold:
        return "PINCH_MIDDLE"

    return "NONE"


def stable_gesture(current):
    gesture_history.append(current)
    if len(gesture_history) > GESTURE_BUFFER:
        gesture_history.pop(0)
    if len(gesture_history) == GESTURE_BUFFER and all(
        g == current for g in gesture_history
    ):
        return current
    return None


# Adaptive smoothing with edge dead zone
def adaptive_smooth(smooth_x, smooth_y, target_x, target_y):
    dist = math.hypot(target_x - smooth_x, target_y - smooth_y)
    t = (dist - VELOCITY_THRESHOLD_LOW) / max(
        VELOCITY_THRESHOLD_HIGH - VELOCITY_THRESHOLD_LOW, 1
    )
    t = max(0.0, min(1.0, t))
    alpha = SMOOTH_MIN + t * (SMOOTH_MAX - SMOOTH_MIN)

    edge_x = min(smooth_x / sw, 1.0 - smooth_x / sw)
    edge_y = min(smooth_y / sh, 1.0 - smooth_y / sh)
    edge_proximity = min(edge_x, edge_y)
    if edge_proximity < EDGE_ZONE:
        dampen = edge_proximity / EDGE_ZONE
        alpha *= EDGE_DAMPING + (1.0 - EDGE_DAMPING) * dampen

    return (
        smooth_x + (target_x - smooth_x) * alpha,
        smooth_y + (target_y - smooth_y) * alpha,
    )


# Mission Control
def toggle_mission_control():
    global last_mission_control_time
    now = time.time()
    if now - last_mission_control_time < MISSION_CONTROL_COOLDOWN:
        return
    subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to key code 160']
    )
    last_mission_control_time = now


# Gesture handler
def handle_gesture(gesture, last_gesture):
    global is_dragging, pinch_start_time

    now = time.time()

    if gesture == "PINCH_INDEX":
        if pinch_start_time is None:
            pinch_start_time = now
        if not is_dragging and (now - pinch_start_time) >= LONG_PINCH_DURATION:
            mouse.press(Button.left)
            is_dragging = True
    else:
        if pinch_start_time is not None:
            held = now - pinch_start_time
            pinch_start_time = None
            if is_dragging:
                mouse.release(Button.left)
                is_dragging = False
            elif held < LONG_PINCH_DURATION and gesture != last_gesture:
                mouse.click(Button.left)

    if gesture == last_gesture:
        return

    if gesture == "PINCH_MIDDLE":
        mouse.click(Button.right)

    elif gesture == "OPEN":
        toggle_mission_control()


# Main loop
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
            raw_gesture = detect_gesture(hand_landmarks)
            gesture = stable_gesture(raw_gesture)

            handle_gesture(gesture, last_gesture)
            if gesture is not None:
                last_gesture = gesture

            anchor_x = sum(hand_landmarks[i].x for i in PALM_IDS) / len(PALM_IDS)
            anchor_y = sum(hand_landmarks[i].y for i in PALM_IDS) / len(PALM_IDS)

            raw_x = (anchor_x - ZONE_LEFT) / (ZONE_RIGHT - ZONE_LEFT)
            raw_y = (anchor_y - ZONE_TOP) / (ZONE_BOTTOM - ZONE_TOP)
            raw_x = max(0.0, min(1.0, raw_x))
            raw_y = max(0.0, min(1.0, raw_y))

            target_x = int(raw_x * sw)
            target_y = int(raw_y * sh)

            smooth_x, smooth_y = adaptive_smooth(smooth_x, smooth_y, target_x, target_y)
            mouse.position = (int(smooth_x), int(smooth_y))

            if args.debug:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(img, pts[a], pts[b], (0, 255, 255), 2)
                for i, (cx, cy) in enumerate(pts):
                    color = (0, 0, 255) if i in FINGERTIP_IDS else (255, 255, 255)
                    r = 15 if i in FINGERTIP_IDS else 10
                    cv2.circle(img, (cx, cy), r, color, cv2.FILLED)

                display_gesture = "DRAG" if is_dragging else (gesture or "")
                if display_gesture:
                    cv2.putText(
                        img,
                        display_gesture,
                        (10, 110),
                        cv2.FONT_HERSHEY_PLAIN,
                        3,
                        (0, 255, 0),
                        3,
                    )

                scores = finger_straightness(hand_landmarks)
                score_str = " ".join(f"{s:.2f}" for s in scores)
                cv2.putText(
                    img,
                    f"str: {score_str}",
                    (10, 200),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (200, 200, 0),
                    2,
                )

                pw = palm_width(hand_landmarks)
                pd = normalised_pinch_distance(hand_landmarks, 4, 8)
                rot = hand_rotation_factor(hand_landmarks)
                thr = effective_pinch_threshold(hand_landmarks)
                cv2.putText(
                    img,
                    f"pinch:{pd:.2f} thr:{thr:.2f} rot:{rot:.2f}",
                    (10, 160),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 200, 0),
                    2,
                )

        if not result.hand_landmarks:
            if is_dragging:
                mouse.release(Button.left)
                is_dragging = False
            if pinch_start_time is not None:
                pinch_start_time = None
            last_gesture = None

        if args.debug:
            zone_x1, zone_y1 = int(ZONE_LEFT * w), int(ZONE_TOP * h)
            zone_x2, zone_y2 = int(ZONE_RIGHT * w), int(ZONE_BOTTOM * h)
            cv2.rectangle(img, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 0), 2)

            cTime = time.time()
            fps = 1 / (cTime - pTime) if pTime else 0
            pTime = cTime
            cv2.putText(
                img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3
            )
            cv2.imshow("Debug", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

cap.release()
if args.debug:
    cv2.destroyAllWindows()
