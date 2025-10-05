"""
AI Fitness Agent (OpenCV + MediaPipe)
Supports:
 - Squat rep counting + simple posture warnings (knee over toe, torso too forward)
 - Push-up rep counting + simple posture warnings (hips sagging / too high)

Usage:
    python ai_fitness_agent.py --exercise squat
    python ai_fitness_agent.py --exercise pushup
Optional args:
    --camera 0       # camera index
    --display True   # set False to turn off display overlays
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import argparse
import time

# ----------------------------
# Utilities: Angle & helpers
# ----------------------------
def angle_between_points(a, b, c):
    """
    Calculate the angle at point b given three points a, b, c.
    Each point is (x, y).
    Returns angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    # cos theta = (ba . bc) / (|ba||bc|)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def get_landmark_coords(landmarks, idx, frame_w, frame_h):
    lm = landmarks[idx]
    return int(lm.x * frame_w), int(lm.y * frame_h)

# ----------------------------
# Pose Detector Setup
# ----------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----------------------------
# Exercise classes
# ----------------------------
class SquatCounter:
    def __init__(self, down_ang_thresh=95, up_ang_thresh=165):
        # knee angle thresholds (knee formed by hip-knee-ankle)
        self.down_thresh = down_ang_thresh
        self.up_thresh = up_ang_thresh
        self.state = "up"  # or "down"
        self.count = 0
        self.last_rep_time = 0

    def process(self, landmarks, w, h):
        # landmarks indexes for mediapipe
        # left side
        hip_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, w, h)
        knee_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value, w, h)
        ankle_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value, w, h)
        # right side
        hip_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, w, h)
        knee_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value, w, h)
        ankle_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value, w, h)

        # compute average knee angle between left & right
        knee_angle_l = angle_between_points(hip_l, knee_l, ankle_l)
        knee_angle_r = angle_between_points(hip_r, knee_r, ankle_r)
        knee_angle = (knee_angle_l + knee_angle_r) / 2.0

        # Rep counting state machine
        feedback = []
        t = time.time()
        if knee_angle < self.down_thresh and self.state == "up":
            # moved down
            self.state = "down"
        if knee_angle > self.up_thresh and self.state == "down":
            # completed rep (up after down)
            # debounce: ensure at least 0.5s between reps
            if t - self.last_rep_time > 0.5:
                self.count += 1
                self.last_rep_time = t
                self.state = "up"

        # Posture checks:
        # 1) knee over toe: check horizontal relation between knee and ankle
        # if knee x significantly ahead of ankle x (in camera view) - warn
        # Do this separately per side and report if either side is problematic
        if abs(knee_l[0] - ankle_l[0]) > 0:
            if knee_l[0] - ankle_l[0] > 30:  # knee too forward to the right (camera coords)
                feedback.append("Left knee ahead of toe")
        if abs(knee_r[0] - ankle_r[0]) > 0:
            if ankle_r[0] - knee_r[0] > 30:  # knee too forward to the left (camera coords)
                feedback.append("Right knee ahead of toe")

        # 2) torso lean: measure angle of shoulder-hip relative to vertical/horizontal
        shoulder_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
        shoulder_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
        shoulder = ((shoulder_l[0] + shoulder_r[0])//2, (shoulder_l[1] + shoulder_r[1])//2)
        hip = ((hip_l[0] + hip_r[0])//2, (hip_l[1] + hip_r[1])//2)
        # torso vector
        torso_angle = angle_between_points( (shoulder[0], shoulder[1]-10), shoulder, hip )
        # If torso angle deviates much from ~90 degrees (vertical) or shows forward lean (small angle),
        # we can warn. For simplicity, if torso_angle < 75 -> too forward
        if torso_angle < 75:
            feedback.append("Torso too forward - keep chest up")

        return {
            "knee_angle": knee_angle,
            "count": self.count,
            "state": self.state,
            "feedback": feedback,
            "keypoints": {
                "hip_l": hip_l, "knee_l": knee_l, "ankle_l": ankle_l,
                "hip_r": hip_r, "knee_r": knee_r, "ankle_r": ankle_r,
                "shoulder": shoulder, "hip_mid": hip
            }
        }

class PushupCounter:
    def __init__(self, down_ang_thresh=90, up_ang_thresh=160):
        # elbow angle thresholds (shoulder-elbow-wrist)
        self.down_thresh = down_ang_thresh
        self.up_thresh = up_ang_thresh
        self.state = "up"
        self.count = 0
        self.last_rep_time = 0

    def process(self, landmarks, w, h):
        shoulder_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
        elbow_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value, w, h)
        wrist_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value, w, h)

        shoulder_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
        elbow_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value, w, h)
        wrist_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value, w, h)

        # average elbow angles
        elbow_angle_l = angle_between_points(shoulder_l, elbow_l, wrist_l)
        elbow_angle_r = angle_between_points(shoulder_r, elbow_r, wrist_r)
        elbow_angle = (elbow_angle_l + elbow_angle_r) / 2.0

        # state machine
        feedback = []
        t = time.time()
        if elbow_angle < self.down_thresh and self.state == "up":
            self.state = "down"
        if elbow_angle > self.up_thresh and self.state == "down":
            if t - self.last_rep_time > 0.4:
                self.count += 1
                self.last_rep_time = t
                self.state = "up"

        # Posture: check body alignment (shoulder-hip-ankle roughly collinear vertically)
        hip_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, w, h)
        hip_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, w, h)
        ankle_l = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value, w, h)
        ankle_r = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value, w, h)

        shoulder_mid = ((shoulder_l[0] + shoulder_r[0])//2, (shoulder_l[1] + shoulder_r[1])//2)
        hip_mid = ((hip_l[0] + hip_r[0])//2, (hip_l[1] + hip_r[1])//2)
        ankle_mid = ((ankle_l[0] + ankle_r[0])//2, (ankle_l[1] + ankle_r[1])//2)

        # compute vertical alignment error: distance from hip to line shoulder-ankle
        def point_line_distance(p, a, b):
            # distance from p to line ab
            p = np.array(p); a = np.array(a); b = np.array(b)
            if np.linalg.norm(b - a) < 1e-6:
                return np.linalg.norm(p - a)
            return np.abs(np.cross(b-a, a-p)) / (np.linalg.norm(b-a) + 1e-8)

        alignment_err = point_line_distance(hip_mid, shoulder_mid, ankle_mid)
        # normalize by frame height roughly (higher error is worse)
        if alignment_err > 30:
            feedback.append("Hips misaligned - keep body straight")

        return {
            "elbow_angle": elbow_angle,
            "count": self.count,
            "state": self.state,
            "feedback": feedback,
            "keypoints": {
                "shoulder": shoulder_mid, "hip_mid": hip_mid, "ankle_mid": ankle_mid
            }
        }

# ----------------------------
# Main loop
# ----------------------------
def main(exercise="squat", camera_idx=0, display=True):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if exercise.lower() == "squat":
        counter = SquatCounter()
    elif exercise.lower() == "pushup":
        counter = PushupCounter()
    else:
        print("Unsupported exercise. Use 'squat' or 'pushup'.")
        return

    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # mirror for natural interaction
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose detection
        results = pose_model.process(rgb)
        feedback_messages = []
        keypoints = {}
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(200,200,200), thickness=1))
            landmarks = results.pose_landmarks.landmark
            out = counter.process(landmarks, w, h)
            keypoints = out.get("keypoints", {})
            feedback_messages = out.get("feedback", [])
            count = out.get("count", 0)
            state = out.get("state", "")

            # overlay rep count & state
            cv2.rectangle(frame, (0,0), (250,70), (0,0,0), -1)
            cv2.putText(frame, f"Exercise: {exercise.title()}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Reps: {count}", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"State: {state}", (130,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

            # draw feedback messages
            for i, msg in enumerate(feedback_messages[:3]):
                cv2.putText(frame, f"⚠️ {msg}", (10, 80 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

            # draw keypoints of interest
            for k, v in keypoints.items():
                if isinstance(v, tuple):
                    cv2.circle(frame, v, 6, (255,0,0), -1)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-8)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        if display:
            cv2.imshow("AI Fitness Agent", frame)
        # press q to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", type=str, default="squat", help="squat or pushup")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--display", type=bool, default=True, help="display overlay window")
    args = parser.parse_args()

    main(exercise=args.exercise, camera_idx=args.camera, display=args.display)
