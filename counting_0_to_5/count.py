"""
Name: Ava Kirkland
Date: 3/15/2026
Description: Builds on experiment2.py — adds two rule-based custom gestures:
               - Finger count (0–5), displayed near the wrist on each hand
               - Gun shape (index + thumb extended, rest curled)
             Rule-based gestures override the built-in model label when detected.
             draw_secondary_gesture removed (model only returns top gesture).
"""

import cv2
import mediapipe as mp
import time
import math

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = "gesture_recognizer.task"
CAMERA_ID       = 0
FONT            = cv2.FONT_HERSHEY_SIMPLEX
PINCH_THRESHOLD = 0.25
THUMB_ANGLE_THRESHOLD = 25  # degrees — experimentally determined to match visual thumb extension

# ── Gesture Style Map ─────────────────────────────────────────────────────────
# BGR colors. To add a new gesture: one line here, display updates everywhere.
GESTURE_STYLE = {
    "None":        ("None",       (120, 120, 120)),
    "Unknown":     ("?",          (100, 100, 100)),
    "Closed_Fist": ("Fist",       (50,  50,  220)),
    "Open_Palm":   ("Open Hand",  (50,  200,  50)),
    "Pointing_Up": ("Point Up",   (220, 180,  50)),
    "Thumb_Down":  ("Thumb Down", (30,   30, 200)),
    "Thumb_Up":    ("Thumb Up",   (50,  220, 150)),
    "Victory":     ("Peace",      (220,  50, 180)),
    "ILoveYou":    ("ILY",        (220, 130,  50)),
    "Pinch":       ("Pinch",      (0,   200, 255)),
    "Gun":         ("Gun",        (0,   100, 255)),  # orange — custom gesture
}

def get_gesture_style(gesture_name):
    """Falls back to raw name in white for any unrecognized gesture."""
    return GESTURE_STYLE.get(gesture_name, (gesture_name, (255, 255, 255)))


# ── Landmark drawing (unchanged from experiment2.py) ─────────────────────────
#has colors for the dot and connections for each finger, and the palm
FINGER_GROUPS = [
    ([(0,1),(0,5),(0,9),(0,13),(0,17),(5,9),(9,13),(13,17)],
     (180, 180, 180), (0, 0, 220)),
    ([(1,2),(2,3),(3,4)],          (120, 190, 210), (120, 190, 210)),
    ([(5,6),(6,7),(7,8)],          (180,  60, 150), (180,  60, 150)),
    ([(9,10),(10,11),(11,12)],     (0,   210, 220), (0,   210, 220)),
    ([(13,14),(14,15),(15,16)],    (0,   200,  80), (0,   200,  80)),
    ([(17,18),(18,19),(19,20)],    (220, 100,   0), (220, 100,   0)),
]

def draw_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for connections, line_color, dot_color in FINGER_GROUPS:
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], line_color, 2)
            cv2.circle(frame, pts[a], 6, dot_color,       -1)
            cv2.circle(frame, pts[a], 6, (255, 255, 255),  1)
            cv2.circle(frame, pts[b], 6, dot_color,       -1)
            cv2.circle(frame, pts[b], 6, (255, 255, 255),  1)


# ── Rule-based gesture detection (new in experiment3.py) ─────────────────────

# Landmark index reference used below:
#
#   Wrist = 0
#
#   Edge points to inner hand (for drawing connections):
#   Finger   TIP   DIP   PIP   MCP
#   Thumb     4     3     2     1
#   Index     8     7     6     5
#   Middle   12    11    10     9
#   Ring     16    15    14    13
#   Pinky    20    19    18    17
#
# "Extended" = tip is further from wrist than PIP (middle knuckle).
# We use PIP rather than MCP (base knuckle) because PIP gives a cleaner
# signal — a finger bent at the base but straight at the tip still looks
# extended visually, but PIP catches it as curled.

# Pairs of (tip_index, pip_index) for the four non-thumb fingers
FINGER_TIP_PIP = [
    (8,  6),   # Index
    (12, 10),  # Middle
    (16, 14),  # Ring
    (20, 18),  # Pinky
]

def _dist(lm1, lm2):
    """Euclidean distance between two landmarks (in normalized 0–1 space)."""
    return math.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2)

def _is_finger_extended(tip, pip, wrist):
    """
    True if tip is further from wrist than pip is.
    Using wrist as the reference point means the comparison is consistent
    regardless of where in the frame the hand is sitting.
    """
    return _dist(tip, wrist) > _dist(pip, wrist)

#TODO Issue #1: thumb detection is less reliable than fingers — needs a different approach, expectually when hand is flipped where the palm is facing the user
def _is_thumb_extended(hand_landmarks, is_right_hand):
    """
    The thumb extends sideways, not upward, so the vertical distance trick
    used for fingers doesn't work reliably.

    Instead: compare the x-position of the thumb tip (4) vs the thumb IP
    joint (3, one step back from the tip).

    For a RIGHT hand (after mirror flip): thumb extended = tip is LEFT of IP
    (lower x value, since x=0 is the left edge of the frame).
    For a LEFT hand: thumb extended = tip is RIGHT of IP (higher x).

    Why landmark 3 and not landmark 2 (PIP)?
    The thumb's IP joint (3) is the single bend in the thumb.
    Using landmark 2 (the CMC joint) would include the thumb's natural
    sideways offset and produce false positives.
    """
    # tip = hand_landmarks[4]
    # ip  = hand_landmarks[3]

    # pinky_mcp = hand_landmarks[17]
    # wrist     = hand_landmarks[0]

    # is_thumb_extended = False


    # if is_right_hand:
    #     is_hand_flipped = pinky_mcp.x < wrist.x  # pinky base left of wrist = flipped
    #     if is_hand_flipped:
    #         is_thumb_extended = tip.x > ip.x   # tip further right = extended outward
    #     else:
    #         is_thumb_extended = tip.x < ip.x   # tip further left = extended outward
    #     return is_thumb_extended 
    # else:
    #     is_hand_flipped = pinky_mcp.x > wrist.x  # pinky base right of wrist = flipped
    #     if is_hand_flipped:
    #         is_thumb_extended = tip.x < ip.x   # tip further left = extended outward
    #     else:
    #         is_thumb_extended = tip.x > ip.x   # tip further right = extended outward
    #     return is_thumb_extended
    #experiement 2 of thumb detection, issue with if hand not in ideal upright position.
    # tip   = hand_landmarks[4]
    # mcp   = hand_landmarks[2]
    # wrist = hand_landmarks[0]

    # return _dist(tip, wrist) > _dist(mcp, wrist)
    index_tip = hand_landmarks[8]
    cmc   = hand_landmarks[1]
    thumb_tip   = hand_landmarks[4]

    # Vector A: from CMC back toward wrist
    ax = index_tip.x - cmc.x
    ay = index_tip.y - cmc.y

    # Vector B: from CMC forward toward tip
    bx = thumb_tip.x - cmc.x
    by = thumb_tip.y - cmc.y

    # Dot product and magnitudes
    dot      = ax * bx + ay * by
    mag_a    = math.sqrt(ax**2 + ay**2)
    mag_b    = math.sqrt(bx**2 + by**2)

    # Guard against zero-length vectors (hand barely visible)
    if mag_a < 1e-6 or mag_b < 1e-6:
        return False

    # Clamp to [-1, 1] before acos to avoid math domain errors from
    # floating point noise slightly outside that range
    cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
    angle_deg = math.degrees(math.acos(cos_angle))

    #print(f"thumb angle: {angle_deg:.1f}")
    return angle_deg > THUMB_ANGLE_THRESHOLD


def count_extended_fingers(hand_landmarks, handedness):
    """
    Returns the number of extended fingers (0–5) for one hand.

    handedness: "Left" or "Right" — already mirror-corrected by the caller.
    """
    wrist        = hand_landmarks[0]
    is_right     = (handedness == "Right")
    count        = 0

    # Thumb
    if _is_thumb_extended(hand_landmarks, is_right):
        count += 1

    # Four fingers
    for tip_idx, pip_idx in FINGER_TIP_PIP:
        tip = hand_landmarks[tip_idx]
        pip = hand_landmarks[pip_idx]
        if _is_finger_extended(tip, pip, wrist):
            count += 1

    return count

def is_gun_shape(hand_landmarks, handedness):
    """
    True when ONLY index finger and thumb are extended — the 'gun' shape.

    Why check each finger explicitly instead of just count == 2?
    Count == 2 would also match thumb+pinky or index+middle.
    We need to verify *which* two fingers are up, not just how many.
    """
    wrist    = hand_landmarks[0]
    is_right = (handedness == "Right")

    thumb_up  = _is_thumb_extended(hand_landmarks, is_right)

    index_tip, index_pip   = hand_landmarks[8],  hand_landmarks[6]
    middle_tip, middle_pip = hand_landmarks[12], hand_landmarks[10]
    ring_tip,   ring_pip   = hand_landmarks[16], hand_landmarks[14]
    pinky_tip,  pinky_pip  = hand_landmarks[20], hand_landmarks[18]

    index_up  = _is_finger_extended(index_tip,  index_pip,  wrist)
    middle_up = _is_finger_extended(middle_tip, middle_pip, wrist)
    ring_up   = _is_finger_extended(ring_tip,   ring_pip,   wrist)
    pinky_up  = _is_finger_extended(pinky_tip,  pinky_pip,  wrist)

    is_gun  = thumb_up and index_up and not middle_up and not ring_up and not pinky_up

    return is_gun


# ── Display helpers ───────────────────────────────────────────────────────────

def draw_confidence_bar(frame, score, color, x, y, width=120, height=12):
    """Horizontal bar showing model confidence. Unchanged from experiment2.py."""
    cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 40), -1)
    fill_width = int(width * score)
    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)

def draw_primary_gesture(frame, gesture_name, score, handedness, hand_index):
    """
    Top-left gesture label + confidence bar for one hand.
    score=None skips the bar (used when a rule-based gesture overrides).
    """
    label_text, color = get_gesture_style(gesture_name)
    display = f"{handedness}:  {label_text}"

    y_text = 50 + hand_index * 70
    y_bar  = y_text + 12

    cv2.putText(frame, display, (21, y_text + 1), FONT, 1.1, (0, 0, 0), 3)
    cv2.putText(frame, display, (20, y_text),     FONT, 1.1, color,     2)

    if score is not None:
        draw_confidence_bar(frame, score, color, x=20, y=y_bar, width=130, height=10)
        pct_text = f"{score:.0%}"
        cv2.putText(frame, pct_text, (158, y_bar + 10), FONT, 0.55, color, 1)

def draw_pinch_label(frame, hand_index):
    """Pinch override label. Unchanged from experiment2.py."""
    y_pos = 50 + hand_index * 70 + 48
    cv2.putText(frame, "Pinch", (21, y_pos + 1), FONT, 1.1, (0, 0, 0),     3)
    cv2.putText(frame, "Pinch", (20, y_pos),     FONT, 1.1, (0, 200, 255), 2)

def draw_finger_count(frame, hand_landmarks, count):
    """
    Draw the finger count as a large number near the wrist.

    Why near the wrist and not a fixed corner?
    It follows the hand on screen, so with two hands you always know
    which count belongs to which hand without looking at the label.

    Offset: we shift slightly below and right of landmark 0 (wrist) so the
    number doesn't sit directly on top of the wrist dot.
    """
    h, w = frame.shape[:2]
    wrist = hand_landmarks[0]
    x = int(wrist.x * w) + 15
    y = int(wrist.y * h) + 40

    text  = str(count)
    color = (0, 255, 220)  # bright cyan — distinct from landmark colors

    cv2.putText(frame, text, (x + 1, y + 1), FONT, 2.0, (0, 0, 0), 5)  # shadow
    cv2.putText(frame, text, (x,     y),     FONT, 2.0, color,     3)


# ── Pinch detection (unchanged from experiment2.py) ──────────────────────────



def is_pinching(hand_landmarks):
    thumb_tip   = hand_landmarks[4]
    index_tip   = hand_landmarks[8]
    wrist       = hand_landmarks[0]
    mid_knuckle = hand_landmarks[9]

    pinch_dist = _dist(thumb_tip, index_tip)
    hand_size  = _dist(wrist, mid_knuckle)

    if hand_size < 1e-6:
        return False
    return (pinch_dist / hand_size) < PINCH_THRESHOLD


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
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
        start_time = prev_time # fixed referance point for consisten times
        # VIDEO mode requires strictly increasing timestamps in milliseconds.
        # We derive them from elapsed time since the script started rather than
        # wall-clock time, so the numbers stay small and predictable.
        last_ts_ms = -1



        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb.copy())

            
            # Build a strictly-increasing timestamp.
            # max() against last_ts_ms + 1 guarantees MediaPipe never sees
            # two identical timestamps, even if the system clock doesn't
            # advance between two very fast frames.
            ts_ms      = int((time.time() - start_time) * 1000)
            ts_ms      = max(ts_ms, last_ts_ms + 1)
            last_ts_ms = ts_ms

            result   = recognizer.recognize_for_video(mp_image,ts_ms)
            # ── DIAGNOSTIC — remove after testing ────────────────────────────────────────
            if result.hand_landmarks:
                lm = result.hand_landmarks[0]  # just first hand
                thumb_tip  = lm[4]
                thumb_mcp  = lm[2]
                index_tip  = lm[8]

                # Vector A: MCP → thumb tip
                ax, ay, az = thumb_tip.x - thumb_mcp.x, thumb_tip.y - thumb_mcp.y, thumb_tip.z - thumb_mcp.z
                # Vector B: MCP → index tip
                bx, by, bz = index_tip.x - thumb_mcp.x, index_tip.y - thumb_mcp.y, index_tip.z - thumb_mcp.z

                dot   = ax*bx + ay*by + az*bz
                mag_a = math.sqrt(ax**2 + ay**2 + az**2)
                mag_b = math.sqrt(bx**2 + by**2 + bz**2)

                if mag_a > 1e-6 and mag_b > 1e-6:
                    cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
                    angle_deg = math.degrees(math.acos(cos_angle))
                    print(f"thumb-MCP-index angle: {angle_deg:.1f}")
            # ── END DIAGNOSTIC ────────────────────────────────────────────────────────────
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                draw_landmarks(frame, hand_landmarks)

                # Mirror-correct handedness to match flipped camera view
                raw_hand   = result.handedness[i][0]
                handedness = "Right" if raw_hand.category_name == "Left" else "Left"

                # ── Rule-based detections ──────────────────────────────────
                finger_count = count_extended_fingers(hand_landmarks, handedness)
                gun          = is_gun_shape(hand_landmarks, handedness)
                pinch        = is_pinching(hand_landmarks)

                # ── Decide what to show in the top-left label ──────────────
                # Priority: Gun > Pinch > built-in model result
                # Why this order? Our custom rules are more specific than the
                # model — "Gun" requires an exact finger combination that the
                # built-in model has no label for at all.
                if gun:
                    draw_primary_gesture(frame, "Gun", None, handedness, i)
                elif pinch:
                    draw_primary_gesture(frame, "Pinch", None, handedness, i)
                elif result.gestures and i < len(result.gestures):
                    top = result.gestures[i][0]
                    draw_primary_gesture(frame, top.category_name, top.score, handedness, i)

                # Finger count always shows — it's informational, not a label override
                draw_finger_count(frame, hand_landmarks, finger_count)

            # FPS (top-right, unchanged)
            cur_time = time.time()
            fps      = 1.0 / max(cur_time - prev_time, 1e-6)
            prev_time = cur_time
            fps_text  = f"FPS: {fps:.0f}"
            cv2.putText(frame, fps_text, (frame.shape[1] - 130, 30), FONT, 0.8, (0,0,0),       2)
            cv2.putText(frame, fps_text, (frame.shape[1] - 130, 30), FONT, 0.8, (200,200,200), 1)

            cv2.imshow("MediaPipe Gesture Recognizer", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()