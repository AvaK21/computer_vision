"""
Name: Ava Kirkland
Date: 3/15/22026
Description: This script demonstrates how to use MediaPipe's Gesture Recognizer in Python.

Utlizes broswer claude code to generate the code."""

import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "gesture_recognizer.task"   # must sit next to this script
CAMERA_ID  = 0
FONT       = cv2.FONT_HERSHEY_SIMPLEX

# Landmark connections for drawing the hand skeleton
FINGER_GROUPS = [
    # palm / wrist connections — gray lines, red dots
    ([(0,1),(0,5),(0,9),(0,13),(0,17),(5,9),(9,13),(13,17)],
     (180, 180, 180), (0, 0, 220)),
    # thumb — tan lines & dots
    ([(1,2),(2,3),(3,4)],
     (120, 190, 210), (120, 190, 210)),
    # index — purple lines & dots
    ([(5,6),(6,7),(7,8)],
     (180, 60, 150), (180, 60, 150)),
    # middle — yellow lines & dots
    ([(9,10),(10,11),(11,12)],
     (0, 210, 220), (0, 210, 220)),
    # ring — green lines & dots
    ([(13,14),(14,15),(15,16)],
     (0, 200, 80), (0, 200, 80)),
    # pinky — blue lines & dots
    ([(17,18),(18,19),(19,20)],
     (220, 100, 0), (220, 100, 0)),
]

# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for connections, line_color, dot_color in FINGER_GROUPS:
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], line_color, 2)
        for a, b in connections:
            for idx in (a, b):
                cv2.circle(frame, pts[idx], 6, dot_color, -1)
                cv2.circle(frame, pts[idx], 6, (255, 255, 255), 1)


def draw_gesture_label(frame, gesture_name, score, hand_index):
    label = f"{gesture_name}  ({score:.0%})"
    y_pos = 50 + hand_index * 45
    # shadow
    cv2.putText(frame, label, (21, y_pos + 1), FONT, 1.1, (0, 0, 0),     3)
    # text
    cv2.putText(frame, label, (20, y_pos),     FONT, 1.1, (0, 255, 160), 2)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Build the recognizer (IMAGE mode = we push frames manually)
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_ID}")

    print("Press  Q  to quit.")


    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        prev_time = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)          # mirror for natural feel

            # Convert BGR → RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = recognizer.recognize(mp_image)

            # Draw landmarks + labels for every detected hand
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                draw_landmarks(frame, hand_landmarks)

                if result.gestures and i < len(result.gestures):
                    top = result.gestures[i][0]          # highest-confidence gesture
                    draw_gesture_label(frame, top.category_name, top.score, i)

            # FPS counter (top-right)
            cur_time = time.time()
            fps = 1.0 / max(cur_time - prev_time, 1e-6)
            prev_time = cur_time
            fps_text = f"FPS: {fps:.0f}"
            cv2.putText(frame, fps_text,
                        (frame.shape[1] - 130, 30), FONT, 0.8, (0, 0, 0),     2)
            cv2.putText(frame, fps_text,
                        (frame.shape[1] - 130, 30), FONT, 0.8, (200, 200, 200), 1)

            cv2.imshow("MediaPipe Gesture Recognizer", frame)


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()