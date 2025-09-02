# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 13:14:52 2025

@author: Quantum Mark 3
"""

#!/usr/bin/env python3
# Hand Gesture Controller - MediaPipe + OpenCV + PyAutoGUI
# Controls: 
#  - Open Palm  -> Play/Pause (Space)
#  - Victory ✌️ -> Next (Right Arrow)
#  - Index Only -> Previous (Left Arrow)
#  - Thumb Up   -> Volume Up
#  - Thumb Down -> Volume Down
# Press 'q' to quit.

import cv2
import numpy as np
import time
import pyautogui

# Reduce pyautogui fails on high-DPI / fail-safes
pyautogui.FAILSAFE = False

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("Please install mediapipe: pip install mediapipe")

MIRROR = True  # Set False if you prefer non-mirrored preview

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# Finger indices from MediaPipe
TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]  # for thumb we use 3 (IP), others are PIP

# Gesture firing parameters
TRIGGER_HOLD_SECONDS = 0.30   # gesture must persist this long
TRIGGER_COOLDOWN_SECONDS = 1.00  # after firing, wait this long

last_gesture = None
gesture_start_t = 0.0
last_fire_t = 0.0

def fingers_up(handedness_label, lm):
    """
    Return a list of 5 booleans [thumb, index, middle, ring, pinky]
    based on landmark positions.
    For non-rotated hand in camera: 
      - For non-thumb fingers: tip.y < pip.y means 'up'
      - For thumb: compare x relative to IP joint and handedness
    """
    # Convert to np arrays for convenience (normalized coordinates)
    # lm[i].x/y in [0,1] relative to image width/height
    thumb_tip = lm[TIP_IDS[0]]
    thumb_ip  = lm[PIP_IDS[0]]

    # Heuristic: if Right hand, thumb is on left side of its IP in image (x smaller when up).
    # If Left hand, thumb is on right side of its IP (x greater when up).
    if handedness_label == 'Right':
        thumb_is_up = thumb_tip.x < thumb_ip.x - 0.02
    else:
        thumb_is_up = thumb_tip.x > thumb_ip.x + 0.02

    fingers = [thumb_is_up]

    for tip_id, pip_id in zip(TIP_IDS[1:], PIP_IDS[1:]):
        tip = lm[tip_id]
        pip = lm[pip_id]
        fingers.append(tip.y < pip.y - 0.02)  # 0.02 margin to reduce flicker

    return fingers  # [thumb, index, middle, ring, pinky]

def classify_gesture(handedness_label, lm):
    """
    Simple rule-based gestures:
      - Open Palm (all up) -> PLAY_PAUSE
      - Victory (index+middle up only) -> NEXT
      - Index Only -> PREV
      - Thumb Up -> VOL_UP
      - Thumb Down -> VOL_DOWN
    Returns (gesture_name, display_name)
    """
    f = fingers_up(handedness_label, lm)
    thumb, index, middle, ring, pinky = f
    up_count = sum(f)

    # Compute rough thumb direction using wrist->thumb_tip vector
    wrist = lm[0]
    ttip = lm[TIP_IDS[0]]
    dy = (ttip.y - wrist.y)  # positive if thumb tip lower than wrist in image coords

    # Thumb Up/Down (only thumb up, others down; direction by dy)
    if (thumb and not index and not middle and not ring and not pinky):
        if dy < -0.05:
            return ("VOL_UP", "Thumb Up → Volume Up")
        elif dy > 0.05:
            return ("VOL_DOWN", "Thumb Down → Volume Down")

    # Open palm: all fingers up
    if up_count == 5:
        return ("PLAY_PAUSE", "Open Palm → Play/Pause")

    # Victory: only index & middle up
    if index and middle and not ring and not pinky and not thumb:
        return ("NEXT", "Victory ✌️ → Next")

    # Index only up
    if index and not middle and not ring and not pinky and not thumb:
        return ("PREV", "Index Only ☝️ → Previous")

    # Nothing recognized (fist or mixed)
    return (None, "")

def maybe_fire_action(gesture_code):
    """
    Fires mapped hotkeys with cooldown.
    """
    global last_fire_t
    now = time.time()
    if now - last_fire_t < TRIGGER_COOLDOWN_SECONDS:
        return  # still cooling

    if gesture_code == "PLAY_PAUSE":
        pyautogui.press('space')
    elif gesture_code == "NEXT":
        pyautogui.press('right')
    elif gesture_code == "PREV":
        pyautogui.press('left')
    elif gesture_code == "VOL_UP":
        # Cross-platform: try volumeup key, fallback to OS shortcuts if unsupported
        try:
            pyautogui.press('volumeup')
        except Exception:
            pyautogui.hotkey('ctrl', 'up')
    elif gesture_code == "VOL_DOWN":
        try:
            pyautogui.press('volumedown')
        except Exception:
            pyautogui.hotkey('ctrl', 'down')
    else:
        return

    last_fire_t = now

def draw_hud(frame, gesture_text):
    h, w = frame.shape[:2]
    if gesture_text:
        cv2.rectangle(frame, (10, 10), (10 + 440, 80), (0, 0, 0), -1)
        cv2.putText(frame, "Gesture:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, gesture_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,255,200), 2, cv2.LINE_AA)

    help_lines = [
        "q: Quit | Gestures:",
        "Open Palm = Space (Play/Pause)",
        "Victory ✌️ = Right (Next)",
        "Index Only = Left (Previous)",
        "Thumb Up = Volume Up | Thumb Down = Volume Down"
    ]
    y = h - 100
    cv2.rectangle(frame, (10, y-30), (420, h-10), (0,0,0), -1)
    for i, line in enumerate(help_lines):
        cv2.putText(frame, line, (20, y + i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return frame

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    cv2.namedWindow("Hand Gesture Controller", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Gesture Controller", 1000, 600)

    global last_gesture, gesture_start_t

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if MIRROR:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture_code, gesture_display = (None, "")

        if result.multi_hand_landmarks and result.multi_handedness:
            hand_lms = result.multi_hand_landmarks[0]
            handed_label = result.multi_handedness[0].classification[0].label  # "Left" or "Right"

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(128,255,128), thickness=2),
            )

            lm = hand_lms.landmark  # list of 21 normalized landmarks

            gesture_code, gesture_display = classify_gesture(handed_label, lm)

            # Temporal smoothing: require gesture to hold for TRIGGER_HOLD_SECONDS
            now = time.time()
            if gesture_code is None:
                last_gesture = None
                gesture_start_t = 0.0
            else:
                if last_gesture != gesture_code:
                    last_gesture = gesture_code
                    gesture_start_t = now
                else:
                    held = now - gesture_start_t
                    if held >= TRIGGER_HOLD_SECONDS:
                        maybe_fire_action(gesture_code)
                        # reset so user must lift and show again
                        last_gesture = None
                        gesture_start_t = 0.0

        frame = draw_hud(frame, gesture_display)
        cv2.imshow("Hand Gesture Controller", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
