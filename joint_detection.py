import cv2
import mediapipe as mp
import math
import argparse

# modern Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

INTEREST_LANDMARKS = [4, 8, 12, 16, 20] # all fingertip joints
WRIST_LANDMARK = 0
MODEL_PATH = "hand_landmarker.task"

def calculate_distance(p1, p2): # calculate euclidean distance between index finger and reference point (determines z-value)
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def ema(prev, x, alpha=0.35):
    return x if prev is None else (alpha * x + (1 - alpha) * prev)

def main():
    ap = argparse.ArgumentParser() # adding prerecorded video options
    ap.add_argument("--video", type=str, default=None, help="Path to a video file (e.g., iPhone .mp4)")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index if not using --video")
    ap.add_argument("--save", type=str, default=None, help="Optional output path to save annotated video")
    args = ap.parse_args()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.video) if args.video else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0
    dt_frame = 1.0 / fps

    writer = None
    if args.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    print("Running MediaPipe on:", args.video if args.video else f"webcam {args.camera}")
    print("Press 'q' to quit.")

    p_prev = None # snapshot positions of previous point
    v_ema = None
    frame_idx = 0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, image = cap.read() # success is boolean if image was taken or not, image is the actual snapshot
            if not success:
                break

            image = cv2.flip(image, 1)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            timestamp_ms = int((frame_idx / fps) * 1000)
            frame_idx += 1

            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if detection_result.hand_landmarks:
                h, w, _ = image.shape # height, width, and _ because color doesn't matter

                hand_landmarks = detection_result.hand_landmarks[0]

                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

                wrist = hand_landmarks[WRIST_LANDMARK]
                index_tip = hand_landmarks[8]

                dist_index_wrist = calculate_distance(index_tip, wrist)

                current_pos = (index_tip.x, index_tip.y, index_tip.z)
                velocity = 0.0
                if p_prev is not None:
                    dx = current_pos[0] - p_prev[0]
                    dy = current_pos[1] - p_prev[1]
                    dz = current_pos[2] - p_prev[2]
                    velocity = math.sqrt(dx*dx + dy*dy + dz*dz) / dt_frame

                p_prev = current_pos
                v_ema = ema(v_ema, velocity, alpha=0.35)

                cx_w, cy_w = int(wrist.x * w), int(wrist.y * h)
                cx_i, cy_i = int(index_tip.x * w), int(index_tip.y * h)
                cv2.line(image, (cx_w, cy_w), (cx_i, cy_i), (0, 255, 255), 2)

                stats = [
                    f"t = {timestamp_ms/1000:.3f}s",
                    f"FPS = {fps:.1f}",
                    f"Index-Wrist Dist: {dist_index_wrist:.3f}",
                    f"Index Tip Z: {index_tip.z:.3f}",
                    f"Velocity: {velocity:.3f} (raw)",
                    f"Velocity: {0.0 if v_ema is None else v_ema:.3f} (EMA)",
                ]
                for i, text in enumerate(stats):
                    cv2.putText(image, text, (10, 30 + i * 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.imshow("MediaPipe Diagnostic (Video/File)", image)
            if writer:
                writer.write(image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
