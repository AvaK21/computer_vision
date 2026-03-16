"""
Name: Ava Kirkland
Date: 3/15/2026
Description: Gesture recognition using MediaPipe's Gesture Recognizer and OpenCV.
             Detects up to 2 hands via webcam and labels the recognized gesture.
"""

import cv2
import mediapipe as mp
import time
import math

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

#Constants
PINCH_THRESHOLD = 0.25  # Adjust this value to make pinch detection more or less sensitive

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "gesture_recognizer.task"  # .task file must be in the same folder
CAMERA_ID  = 0
FONT       = cv2.FONT_HERSHEY_SIMPLEX

# Each entry: (list of (point_a, point_b) connections, line_color, dot_color)
# Colors are BGR (not RGB) — that's OpenCV's default color format
FINGER_GROUPS = [
    # Palm / wrist base connections
    ([(0,1),(0,5),(0,9),(0,13),(0,17),(5,9),(9,13),(13,17)],
     (180, 180, 180), (0, 0, 220)),
    # Thumb
    ([(1,2),(2,3),(3,4)],
     (120, 190, 210), (120, 190, 210)),
    # Index finger
    ([(5,6),(6,7),(7,8)],
     (180, 60, 150), (180, 60, 150)),
    # Middle finger
    ([(9,10),(10,11),(11,12)],
     (0, 210, 220), (0, 210, 220)),
    # Ring finger
    ([(13,14),(14,15),(15,16)],
     (0, 200, 80), (0, 200, 80)),
    # Pinky
    ([(17,18),(18,19),(19,20)],
     (220, 100, 0), (220, 100, 0)),
]


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_landmarks(frame, hand_landmarks):
    """Draw the hand skeleton (bones + joint dots) onto the frame."""
    h, w = frame.shape[:2]
    # Convert normalized (0.0–1.0) landmark coords to actual pixel positions
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for connections, line_color, dot_color in FINGER_GROUPS:
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], line_color, 2)
            # Draw a white-outlined dot on each joint in this group
            cv2.circle(frame, pts[a], 6, dot_color, -1)
            cv2.circle(frame, pts[a], 6, (255, 255, 255), 1)  # white outline
            cv2.circle(frame, pts[b], 6, dot_color, -1)
            cv2.circle(frame, pts[b], 6, (255, 255, 255), 1)


def draw_gesture_label(frame, gesture_name, score, handedness,  hand_index):
    """Display the gesture name and confidence score on screen."""
    label = f"{handedness}:  {gesture_name}  ({score:.0%})"
    y_pos = 50 + hand_index * 45  # stack labels vertically if 2 hands are present
    cv2.putText(frame, label, (21, y_pos + 1), FONT, 1.1, (0, 0, 0),     3)  # shadow
    cv2.putText(frame, label, (20, y_pos),     FONT, 1.1, (0, 255, 160), 2)  # text

def draw_pinch_label(frame, hand_index):
    """Display a pinch indicator below the gesture label for this hand."""
    y_pos = 95 + hand_index * 45   # sits below draw_gesture_label's 50px starting point
    cv2.putText(frame, "Pinch", (21, y_pos + 1), FONT, 1.1, (0, 0, 0),     3)  # shadow
    cv2.putText(frame, "Pinch", (20, y_pos),     FONT, 1.1, (0, 200, 255), 2)  # yellow-white text
#---Helper functions---------------------------

#If it if finiky, remove the z comoponent and just do 2D distance, z is technicallly on a different scale
#taking out the z made more range was accepted as a pinch
def landmark_distance(lm1,lm2):
    """Calculate the Euclidean distance between two landmarks."""
    return math.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2 )
def is_pinching(hand_landmarks):
    """Return True if thumb tip (4) and index tip (8) are close enough to count as a pinch.
    
    The ratio normalizes for hand size so distance thresholds work at any depth.
    """
    thumb_tip  = hand_landmarks[4]
    index_tip  = hand_landmarks[8]
    wrist      = hand_landmarks[0]
    mid_knuckle = hand_landmarks[9]

    pinch_dist = landmark_distance(thumb_tip, index_tip)
    hand_size  = landmark_distance(wrist, mid_knuckle)

    if hand_size < 1e-6:        # guard against division by zero if hand is barely visible
        return False

    return (pinch_dist / hand_size) < PINCH_THRESHOLD
# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Set up the MediaPipe Gesture Recognizer
    # IMAGE mode means we manually send each frame — as opposed to VIDEO or LIVE_STREAM
    # which handle timing and callbacks differently
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_ID}")

    cv2.namedWindow("MediaPipe Gesture Recognizer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MediaPipe Gesture Recognizer", 1000, 600)
    print("Press  Q  to quit.")

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        prev_time = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # flip horizontally so it feels like a mirror

            # MediaPipe expects RGB; OpenCV gives us BGR — so we convert
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = recognizer.recognize(mp_image)

            # Draw landmarks and gesture label for each detected hand
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                draw_landmarks(frame, hand_landmarks)

                if result.gestures and i < len(result.gestures):
                    top = result.gestures[i][0]  # [0] = highest confidence gesture
                    raw_handedness = result.handedness[i][0]  # get handedness info for this hand

                    #Flip "Left"/"Right" to match the mirrored view
                    handedness = "Right" if raw_handedness.category_name == "Left" else "Left"



                    draw_gesture_label(frame, top.category_name, top.score, handedness, i)
                    if is_pinching(hand_landmarks):
                        draw_pinch_label(frame, i)

            # FPS display (top-right corner)
            cur_time = time.time()
            fps = 1.0 / max(cur_time - prev_time, 1e-6)  # max() prevents division by zero
            prev_time = cur_time
            fps_text = f"FPS: {fps:.0f}"
            cv2.putText(frame, fps_text, (frame.shape[1] - 130, 30), FONT, 0.8, (0,0,0),       2)  # shadow
            cv2.putText(frame, fps_text, (frame.shape[1] - 130, 30), FONT, 0.8, (200,200,200), 1)  # text

            cv2.imshow("MediaPipe Gesture Recognizer", frame)
            

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()