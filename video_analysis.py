from __future__ import annotations

import os
from typing import Dict, Any


def analyze_video(video_path: str) -> Dict[str, Any]:
    """Analyze a video using optional MediaPipe signals with an OpenCV fallback.

    - Uses OpenCV to read metadata and sample frames.
    - If `mediapipe` is available, runs lightweight landmark detectors and
      aggregates simple metrics over sampled frames.
    - Returns a JSON-serializable dictionary.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    import cv2  # local import to avoid global import side effects

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video for analysis")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_seconds = float(frame_count / fps) if fps > 0 else 0.0

    # MediaPipe optional setup
    mp_available = False
    mp_holistic = None
    try:
        import mediapipe as mp  # type: ignore
        mp_available = True
        mp_holistic = mp.solutions.holistic
    except Exception:
        mp_available = False
        mp_holistic = None

    # Sampling strategy: analyze ~1 fps up to a cap to keep it fast
    sample_rate = max(int(fps), 1) if fps > 0 else 1
    max_samples = 120
    samples_taken = 0

    face_detect_frames = 0
    pose_detect_frames = 0
    left_hand_frames = 0
    right_hand_frames = 0
    motion_frames = 0  # fallback motion proxy when mediapipe is unavailable
    posture_score_sum = 0.0

    # Prepare holistic if available
    holistic = None
    if mp_available and mp_holistic is not None:
        try:
            holistic = mp_holistic.Holistic(static_image_mode=False,
                                            model_complexity=0,
                                            enable_segmentation=False,
                                            refine_face_landmarks=False)
        except Exception:
            holistic = None
            mp_available = False

    # OpenCV fallback detectors (used when mediapipe is unavailable)
    face_cascade = None
    try:
        import cv2 as _cv2  # local alias
        face_cascade_path = getattr(_cv2.data, 'haarcascades', '') + 'haarcascade_frontalface_default.xml'
        if os.path.exists(face_cascade_path):
            face_cascade = _cv2.CascadeClassifier(face_cascade_path)
    except Exception:
        face_cascade = None

    prev_gray = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Sample every sample_rate frames
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        # Convert BGR to RGB for ML models
        if mp_available and holistic is not None:
            try:
                import cv2  # local import
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)

                if result.face_landmarks is not None:
                    face_detect_frames += 1
                if result.pose_landmarks is not None:
                    pose_detect_frames += 1
                    # Posture estimate: angle of torso (hips->shoulders) vs vertical
                    pl = result.pose_landmarks.landmark
                    # Landmarks indices per MediaPipe Pose
                    # 11: left_shoulder, 12: right_shoulder, 23: left_hip, 24: right_hip
                    try:
                        ls = pl[11]; rs = pl[12]; lh = pl[23]; rh = pl[24]
                        shoulder_mid_x = (ls.x + rs.x) / 2.0
                        shoulder_mid_y = (ls.y + rs.y) / 2.0
                        hip_mid_x = (lh.x + rh.x) / 2.0
                        hip_mid_y = (lh.y + rh.y) / 2.0
                        dx = shoulder_mid_x - hip_mid_x
                        dy = shoulder_mid_y - hip_mid_y
                        # Angle from vertical in radians
                        import math
                        angle = abs(math.atan2(dx, dy))  # 0 is upright; larger is leaning
                        # Map angle in [0, ~0.7 rad (~40 deg)] to score 1..0
                        score = max(0.0, 1.0 - min(angle, 0.7) / 0.7)
                        posture_score_sum += score
                    except Exception:
                        pass
                if result.left_hand_landmarks is not None:
                    left_hand_frames += 1
                if result.right_hand_landmarks is not None:
                    right_hand_frames += 1
            except Exception:
                # If MediaPipe processing fails mid-way, continue with fallback metrics
                mp_available = False
                holistic = None
        else:
            # OpenCV fallback metrics when MediaPipe is not available
            try:
                import cv2  # local import
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                # Face detection (proxy for eye contact)
                if face_cascade is not None:
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                    if isinstance(faces, tuple):
                        faces_count = 0
                    else:
                        faces_count = len(faces)
                    if faces_count > 0:
                        face_detect_frames += 1

                # Simple motion detection (proxy for hand movement):
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
                    motion_ratio = float(cv2.countNonZero(thresh)) / float(max(1, thresh.size))
                    if motion_ratio > 0.01:
                        motion_frames += 1
                prev_gray = gray
            except Exception:
                pass

        samples_taken += 1
        if samples_taken >= max_samples:
            break

    if holistic is not None:
        try:
            holistic.close()
        except Exception:
            pass
    cap.release()

    # File info
    file_size_bytes = os.path.getsize(video_path)

    # Heuristics
    resolution_label = f"{width}x{height}"
    if width >= 1920 and height >= 1080:
        quality_hint = "High resolution (1080p or above)"
    elif width >= 1280 and height >= 720:
        quality_hint = "HD resolution (>=720p)"
    else:
        quality_hint = "Lower resolution"

    pacing_hint = "Unknown"
    if duration_seconds > 0 and frame_count > 0:
        observed_fps = frame_count / duration_seconds
        if fps > 0 and abs(observed_fps - fps) / max(fps, 1e-6) < 0.2:
            pacing_hint = "Stable frame pacing"
        else:
            pacing_hint = "Variable frame pacing"

    # Compute high-level scores (0..1)
    if samples_taken > 0:
        face_ratio = face_detect_frames / samples_taken
        pose_ratio = pose_detect_frames / samples_taken
        # Use mediapipe hand presence if available; otherwise fall back to motion ratio
        if (left_hand_frames + right_hand_frames) > 0:
            hand_ratio = (left_hand_frames + right_hand_frames) / max(1, 2 * samples_taken)
        else:
            hand_ratio = motion_frames / samples_taken
        posture_avg = posture_score_sum / max(1, pose_detect_frames)
    else:
        face_ratio = 0.0
        pose_ratio = 0.0
        hand_ratio = 0.0
        posture_avg = 0.0

    # Eye contact proxy: fraction of frames with face detected (forward-facing presence)
    eye_contact = float(round(face_ratio, 3))
    # Posture score from pose
    posture = float(round(posture_avg, 3))
    # Hand movement proxy: prefer mediapipe hands; otherwise motion-based ratio
    if samples_taken == 0:
        hand_movement = 0.0
    else:
        if (left_hand_frames + right_hand_frames) > 0:
            hand_movement = float(round((left_hand_frames + right_hand_frames) / (2 * samples_taken), 3))
        else:
            hand_movement = float(round(hand_ratio, 3))

    # Composite confidence as weighted average
    confidence_level = float(round(0.4 * eye_contact + 0.3 * posture + 0.3 * hand_movement, 3))

    mp_summary = {
        "enabled": bool(mp_available),
        "frames_sampled": samples_taken,
        "detections": {
            "face_presence_ratio": float(round(face_ratio, 3)),
            "pose_presence_ratio": float(round(pose_ratio, 3)),
            "hand_presence_ratio": float(round(hand_ratio, 3)),
        },
    }

    return {
        "file": {
            "path": video_path,
            "size_bytes": file_size_bytes,
        },
        "video": {
            "duration_seconds": round(duration_seconds, 2),
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "resolution": {
                "width": width,
                "height": height,
                "label": resolution_label,
            },
        },
        "mediapipe": mp_summary,
        "eye_contact": eye_contact,
        "posture": posture,
        "hand_movement": hand_movement,
        "confidence_level": confidence_level,
        "quick_assessment": {
            "quality_hint": quality_hint,
            "pacing_hint": pacing_hint,
            "notes": [
                "MediaPipe metrics are computed on sampled frames.",
                "Install mediapipe for richer analysis (optional).",
            ],
        },
    }



