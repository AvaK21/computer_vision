"""
Name: Ava Kirkland
Date: 3/15/2026
Description: Builds on experiment.py — same gesture recognition core, but with
             improved display:
               - Each gesture has its own color and short display label
               - A confidence bar shows model certainty visually
               - The #2 ranked gesture is shown smaller below, for debugging
               - Pinch and handedness display are unchanged from experiment.py
"""

import cv2
import mediapipe as mp
import time
import math

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "gesture_recognizer.task"
CAMERA_ID  = 0
FONT       = cv2.FONT_HERSHEY_SIMPLEX

PINCH_THRESHOLD = 0.25

# ── Gesture Style Map ─────────────────────────────────────────────────────────
# Maps MediaPipe's internal gesture name → (short display label, BGR color)
#
# Why a dictionary here instead of inline?
# One place to add/change a gesture's appearance. When you add custom gestures
# later, you just add one line here and the whole display updates automatically.
#
# Color tip: OpenCV uses BGR (Blue, Green, Red), not RGB.
# So (0, 255, 0) is green, (0, 0, 255) is red, (255, 0, 0) is blue.
GESTURE_STYLE = {
    "None":          ("None",        (120, 120, 120)),  # gray
    "Unknown":       ("?",           (100, 100, 100)),  # dark gray
    "Closed_Fist":   ("Fist",        (50,  50,  220)),  # red
    "Open_Palm":     ("Open Hand",   (50,  200, 50)),   # green
    "Pointing_Up":   ("Point Up",    (220, 180, 50)),   # teal-ish
    "Thumb_Down":    ("Thumb Down",  (30,  30,  200)),  # darker red
    "Thumb_Up":      ("Thumb Up",    (50,  220, 150)),  # green-teal
    "Victory":       ("Peace",       (220, 50,  180)),  # purple-pink
    "ILoveYou":      ("ILY",         (220, 130, 50)),   # blue
    "Pinch":         ("Pinch",       (0,   200, 255)),  # yellow — matches draw_pinch_label
}

def get_gesture_style(gesture_name):
    """
    Look up display label and color for a gesture name.
    Falls back to the raw name in white if we don't recognize it.
    This fallback matters: if you add a custom gesture later and forget to
    update GESTURE_STYLE, it still shows up rather than crashing.
    """
    return GESTURE_STYLE.get(gesture_name, (gesture_name, (255, 255, 255)))


# ── Landmark drawing (unchanged from experiment.py) ───────────────────────────
FINGER_GROUPS = [
    ([(0,1),(0,5),(0,9),(0,13),(0,17),(5,9),(9,13),(13,17)],
     (180, 180, 180), (0, 0, 220)),
    ([(1,2),(2,3),(3,4)],   (120, 190, 210), (120, 190, 210)),
    ([(5,6),(6,7),(7,8)],   (180, 60,  150), (180, 60,  150)),
    ([(9,10),(10,11),(11,12)], (0, 210, 220), (0, 210, 220)),
    ([(13,14),(14,15),(15,16)], (0, 200, 80), (0, 200, 80)),
    ([(17,18),(18,19),(19,20)], (220, 100, 0), (220, 100, 0)),
]

def draw_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for connections, line_color, dot_color in FINGER_GROUPS:
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], line_color, 2)
            cv2.circle(frame, pts[a], 6, dot_color, -1)
            cv2.circle(frame, pts[a], 6, (255, 255, 255), 1)
            cv2.circle(frame, pts[b], 6, dot_color, -1)
            cv2.circle(frame, pts[b], 6, (255, 255, 255), 1)


# ── Gesture label display (new in experiment2.py) ─────────────────────────────

def draw_confidence_bar(frame, score, color, x, y, width=120, height=12):
    """
    Draw a small horizontal bar showing model confidence (0.0 to 1.0).

    Why a bar instead of just the number?
    Text like '94%' and '87%' look similar at a glance. A bar makes the
    difference immediately obvious — especially useful when the model is
    uncertain between two gestures.

    x, y is the top-left corner of the bar.
    The bar fills left-to-right proportional to score.
    """
    # Background track (dark rectangle, full width)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)

    # Filled portion (proportional to score)
    fill_width = int(width * score)
    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)

    # Thin white border around the whole bar
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)


def draw_primary_gesture(frame, gesture_name, score, handedness, hand_index):
    """
    Draw the top-ranked gesture label + confidence bar for one hand.

    Layout per hand (stacked vertically if 2 hands):
      [Handedness: Label]   ← large text
      [========    ] 87%    ← confidence bar + percentage
    """
    label_text, color = get_gesture_style(gesture_name)
    display = f"{handedness}:  {label_text}"

    # Vertical position — each hand gets a 70px block
    y_text = 50 + hand_index * 70
    y_bar  = y_text + 12   # bar sits just below the text baseline

    # Label text with drop shadow (shadow first, then colored text on top)
    cv2.putText(frame, display, (21, y_text + 1), FONT, 1.1, (0, 0, 0), 3)
    cv2.putText(frame, display, (20, y_text),     FONT, 1.1, color,     2)

    # Confidence bar
    draw_confidence_bar(frame, score, color, x=20, y=y_bar, width=130, height=10)

    # Percentage text next to the bar
    pct_text = f"{score:.0%}"
    cv2.putText(frame, pct_text, (158, y_bar + 10), FONT, 0.55, color, 1)



def draw_pinch_label(frame, hand_index):
    """Unchanged from experiment.py — pinch overlay for this hand."""
    y_pos = 50 + hand_index * 70 + 48
    cv2.putText(frame, "Pinch", (21, y_pos + 1), FONT, 1.1, (0, 0, 0),     3)
    cv2.putText(frame, "Pinch", (20, y_pos),     FONT, 1.1, (0, 200, 255), 2)


# ── Pinch detection (unchanged from experiment.py) ───────────────────────────

def landmark_distance(lm1, lm2):
    return math.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2)

def is_pinching(hand_landmarks):
    thumb_tip   = hand_landmarks[4]
    index_tip   = hand_landmarks[8]
    wrist       = hand_landmarks[0]
    mid_knuckle = hand_landmarks[9]

    pinch_dist = landmark_distance(thumb_tip, index_tip)
    hand_size  = landmark_distance(wrist, mid_knuckle)

    if hand_size < 1e-6:
        return False
    return (pinch_dist / hand_size) < PINCH_THRESHOLD


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
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

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb.copy())
            result = recognizer.recognize(mp_image)

            for i, hand_landmarks in enumerate(result.hand_landmarks):
                draw_landmarks(frame, hand_landmarks)

                if result.gestures and i < len(result.gestures):
                    gesture_list = result.gestures[i]  # ranked list for this hand
                    top = gesture_list[0]

                    # Flip Left/Right to match the mirrored camera view
                    raw_hand = result.handedness[i][0]
                    handedness = "Right" if raw_hand.category_name == "Left" else "Left"

                    draw_primary_gesture(frame, top.category_name, top.score, handedness, i)

                    if is_pinching(hand_landmarks):
                        draw_pinch_label(frame, i)

            # FPS (top-right, unchanged)
            cur_time = time.time()
            fps = 1.0 / max(cur_time - prev_time, 1e-6)
            prev_time = cur_time
            fps_text = f"FPS: {fps:.0f}"
            cv2.putText(frame, fps_text, (frame.shape[1] - 130, 30), FONT, 0.8, (0,0,0),       2)
            cv2.putText(frame, fps_text, (frame.shape[1] - 130, 30), FONT, 0.8, (200,200,200), 1)

            cv2.imshow("MediaPipe Gesture Recognizer", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()