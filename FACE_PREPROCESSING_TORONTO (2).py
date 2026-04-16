import os
import cv2
import json
import numpy as np
from pathlib import Path

from retinaface import RetinaFace

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIG — only change these paths
# ============================================================
INPUT_FRAMES_DIR         = r"/home/m.byalalu/TORONTO_FRAMES"
OUTPUT_ROOT              = r"/home/m.byalalu/RESULTS"
FACE_LANDMARKER_MODEL_PATH  = r"/home/m.byalalu/face_landmarker.task"
FACE_DETECTOR_MODEL_PATH = r"/home/m.byalalu/blaze_face_short_range.tflite"
# Download face_landmarker.task:
#   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
# Download blaze_face_short_range.tflite:
#   https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite

# ============================================================
# TUNING
# ============================================================
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_MARGIN = 0.25      # 25% expansion on each side of bbox
MAX_CLIPS   = None      # set to integer e.g. 5 to test on first N clips only
PREFER_LARGEST_IF_SCORE_MISSING = True


# ============================================================
# DATASET DISCOVERY
# Your structure:
#   TORONTO_FRAMES/
#       HC/
#           OPEN/
#               N003_02_NSM_OPEN_color/
#                   N003_02_NSM_OPEN_color/   <-- leaf: frames live here
#                       frame_0001.jpg
#       STROKE/
#           OPEN/
#               P001_01_SM_OPEN_color/
#                   P001_01_SM_OPEN_color/
#                       frame_0001.jpg
# ============================================================
def discover_clips(root: str) -> list:
    """
    Walk the entire TORONTO_FRAMES tree and return a list of dicts,
    one per leaf clip folder that contains at least one image.

    Returns:
        [
          {
            "label_str":   "HC" or "STROKE",
            "label_int":   0 (HC) or 1 (STROKE),
            "exercise":    "OPEN",
            "clip_name":   "N003_02_NSM_OPEN_color",
            "clip_dir":    "/full/path/to/leaf/folder",
            "frame_paths": ["/full/path/frame_0001.jpg", ...],
          },
          ...
        ]
    """
    label_map = {"HC": 0, "STROKE": 1}
    clips = []

    root_path = Path(root)
    for label_str, label_int in label_map.items():
        label_dir = root_path / label_str
        if not label_dir.exists():
            print(f"[WARN] Label folder not found: {label_dir}")
            continue

        # Walk every subdirectory under this label
        for dirpath, dirnames, filenames in os.walk(str(label_dir)):
            dirnames.sort()   # reproducible order
            images = sorted([
                os.path.join(dirpath, f)
                for f in filenames
                if Path(f).suffix.lower() in IMAGE_EXTS
            ])
            if not images:
                continue  # not a leaf image folder

            dp = Path(dirpath)
            # Derive exercise name: first subfolder under HC/STROKE
            try:
                rel_parts = dp.relative_to(label_dir).parts
                exercise  = rel_parts[0] if len(rel_parts) >= 1 else "UNKNOWN"
                clip_name = dp.name
            except ValueError:
                exercise  = "UNKNOWN"
                clip_name = dp.name

            clips.append({
                "label_str":   label_str,
                "label_int":   label_int,
                "exercise":    exercise,
                "clip_name":   clip_name,
                "clip_dir":    str(dp),
                "frame_paths": images,
            })

    clips.sort(key=lambda c: (c["label_str"], c["exercise"], c["clip_name"]))
    return clips


# ============================================================
# HELPERS — I/O
# ============================================================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_image(path: str, image_bgr: np.ndarray):
    cv2.imwrite(path, image_bgr)


def blank_image_like(image_bgr: np.ndarray, message: str = "No face detected") -> np.ndarray:
    blank = np.zeros_like(image_bgr)
    cv2.putText(blank, message, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return blank


# ============================================================
# HELPERS — geometry
# ============================================================
def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def expand_bbox_xyxy(x1, y1, x2, y2, img_w, img_h, margin=0.25):
    w  = x2 - x1
    h  = y2 - y1
    mx = int(round(w * margin))
    my = int(round(h * margin))
    return (clamp(x1 - mx, 0, img_w - 1),
            clamp(y1 - my, 0, img_h - 1),
            clamp(x2 + mx, 0, img_w - 1),
            clamp(y2 + my, 0, img_h - 1))


def crop_from_bbox(image_bgr: np.ndarray, bbox_xyxy, margin=0.25):
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = expand_bbox_xyxy(*bbox_xyxy, w, h, margin)
    return image_bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


# ============================================================
# HELPERS — drawing
# ============================================================
def draw_bbox(image_bgr, bbox_xyxy, color=(0, 255, 0), label=None, thickness=2):
    out = image_bgr.copy()
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(out, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def draw_points(image_bgr, points, color=(0, 0, 255), radius=3):
    out = image_bgr.copy()
    for pt in points:
        cv2.circle(out, (int(round(pt[0])), int(round(pt[1]))), radius, color, -1)
    return out


def draw_mediapipe_landmarks(image_bgr, lm_result):
    out = image_bgr.copy()
    if not lm_result or not getattr(lm_result, "face_landmarks", None):
        return out
    if len(lm_result.face_landmarks) == 0:
        return out
    h, w = out.shape[:2]
    for lm in lm_result.face_landmarks[0]:
        x, y = int(round(lm.x * w)), int(round(lm.y * h))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), 1, (0, 255, 255), -1)
    return out


def landmarks_to_original(lm_result, cx1, cy1, cw, ch):
    """Map normalized landmark coords from crop space → original image pixels."""
    if not lm_result or not getattr(lm_result, "face_landmarks", None):
        return []
    if len(lm_result.face_landmarks) == 0:
        return []
    return [(lm.x * cw + cx1, lm.y * ch + cy1)
            for lm in lm_result.face_landmarks[0]]


# ============================================================
# MEDIAPIPE — build once, reuse for all clips
# ============================================================
def build_mp_detector():
    base = python.BaseOptions(model_asset_path=FACE_DETECTOR_MODEL_PATH)
    opts = vision.FaceDetectorOptions(
        base_options=base,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
        min_suppression_threshold=0.3,
    )
    return vision.FaceDetector.create_from_options(opts)


def build_mp_landmarker():
    base = python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH)
    opts = vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def to_mp_image(image_bgr):
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


# ============================================================
# DETECTORS
# ============================================================
def run_retinaface(image_bgr):
    resp = RetinaFace.detect_faces(image_bgr)
    if not resp or not isinstance(resp, dict):
        return None

    best_key, best_score, best_area = None, -1.0, -1.0
    for key, info in resp.items():
        fa = info.get("facial_area")
        if not fa or len(fa) != 4:
            continue
        score = float(info.get("score", -1))
        x1, y1, x2, y2 = fa
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if score > best_score or (score == best_score and area > best_area):
            best_score, best_area, best_key = score, area, key

    if best_key is None:
        return None

    info   = resp[best_key]
    bbox   = tuple(map(int, info["facial_area"]))
    lm     = info.get("landmarks", {})
    points = [lm[k] for k in
              ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]
              if k in lm]
    return {"bbox_xyxy": bbox, "score": float(info.get("score", -1)), "points": points}


def run_mp_detector(detector, image_bgr):
    result = detector.detect(to_mp_image(image_bgr))
    if not result or not getattr(result, "detections", None) or len(result.detections) == 0:
        return None

    best_det, best_score, best_area = None, -1.0, -1.0
    for det in result.detections:
        score = float(det.categories[0].score) if det.categories else -1.0
        bb    = det.bounding_box
        area  = bb.width * bb.height
        if score > best_score or (score == best_score and area > best_area):
            best_score, best_area, best_det = score, area, det

    if best_det is None:
        return None

    bb = best_det.bounding_box
    bbox = (int(bb.origin_x), int(bb.origin_y),
            int(bb.origin_x + bb.width), int(bb.origin_y + bb.height))
    h_, w_ = image_bgr.shape[:2]
    points = [(kp.x * w_, kp.y * h_) for kp in (best_det.keypoints or [])]
    return {"bbox_xyxy": bbox,
            "score": float(best_det.categories[0].score) if best_det.categories else -1.0,
            "points": points}


# ============================================================
# PROCESS ONE FRAME
# Output folder layout per frame:
#   {OUTPUT_ROOT}/{label}/{exercise}/{clip_name}/frame_XXXX/
#       original.jpg
#       retinaface_bbox.jpg
#       mediapipe_bbox.jpg
#       retinaface_crop.jpg
#       mediapipe_crop.jpg
#       mediapipe_landmarks_on_retina_crop.jpg
#       mediapipe_landmarks_on_mp_crop.jpg
# ============================================================
def process_frame(frame_path, out_dir, mp_detector, mp_landmarker):
    image_bgr = cv2.imread(frame_path)
    if image_bgr is None:
        return None

    h, w = image_bgr.shape[:2]
    ensure_dir(out_dir)

    # 1. original
    save_image(os.path.join(out_dir, "original.jpg"), image_bgr)

    row = {"image_w": w, "image_h": h,
           "retinaface_found": False, "mediapipe_found": False}

    # ── RetinaFace branch ──────────────────────────────────────────────────
    retina = run_retinaface(image_bgr)
    if retina is not None:
        row["retinaface_found"] = True
        row["retinaface_score"] = retina["score"]
        x1, y1, x2, y2 = retina["bbox_xyxy"]
        row.update({"rf_x1": x1, "rf_y1": y1, "rf_x2": x2, "rf_y2": y2,
                    "rf_w": x2 - x1, "rf_h": y2 - y1})

        # 2. retinaface_bbox.jpg
        img_rf = draw_bbox(image_bgr, retina["bbox_xyxy"],
                           color=(0, 255, 0),
                           label=f"RetinaFace {retina['score']:.3f}")
        img_rf = draw_points(img_rf, retina["points"], color=(0, 0, 255))
        save_image(os.path.join(out_dir, "retinaface_bbox.jpg"), img_rf)

        # 3. retinaface_crop.jpg
        rf_crop, rf_crop_bbox = crop_from_bbox(image_bgr, retina["bbox_xyxy"], CROP_MARGIN)
        save_image(os.path.join(out_dir, "retinaface_crop.jpg"), rf_crop)

        # 4. mediapipe_landmarks_on_retina_crop.jpg
        lm_rf = mp_landmarker.detect(to_mp_image(rf_crop))
        save_image(os.path.join(out_dir, "mediapipe_landmarks_on_retina_crop.jpg"),
                   draw_mediapipe_landmarks(rf_crop, lm_rf))

        cx1, cy1, cx2, cy2 = rf_crop_bbox
        row["rf_landmark_count"] = len(landmarks_to_original(
            lm_rf, cx1, cy1, cx2 - cx1, cy2 - cy1))
    else:
        # blank placeholders — keeps every frame folder complete
        warn_img  = draw_bbox(image_bgr, (5, 5, max(10, w // 3), max(10, h // 10)),
                              color=(0, 0, 255), label="RetinaFace: no face")
        blank     = blank_image_like(image_bgr, "RetinaFace: no face")
        save_image(os.path.join(out_dir, "retinaface_bbox.jpg"),  warn_img)
        save_image(os.path.join(out_dir, "retinaface_crop.jpg"),  blank)
        save_image(os.path.join(out_dir, "mediapipe_landmarks_on_retina_crop.jpg"), blank)

    # ── MediaPipe branch ───────────────────────────────────────────────────
    mp_det = run_mp_detector(mp_detector, image_bgr)
    if mp_det is not None:
        row["mediapipe_found"] = True
        row["mediapipe_score"] = mp_det["score"]
        x1, y1, x2, y2 = mp_det["bbox_xyxy"]
        row.update({"mp_x1": x1, "mp_y1": y1, "mp_x2": x2, "mp_y2": y2,
                    "mp_w": x2 - x1, "mp_h": y2 - y1})

        # 5. mediapipe_bbox.jpg
        img_mp = draw_bbox(image_bgr, mp_det["bbox_xyxy"],
                           color=(255, 0, 0),
                           label=f"MediaPipe {mp_det['score']:.3f}")
        img_mp = draw_points(img_mp, mp_det["points"], color=(0, 255, 255))
        save_image(os.path.join(out_dir, "mediapipe_bbox.jpg"), img_mp)

        # 6. mediapipe_crop.jpg
        mp_crop, mp_crop_bbox = crop_from_bbox(image_bgr, mp_det["bbox_xyxy"], CROP_MARGIN)
        save_image(os.path.join(out_dir, "mediapipe_crop.jpg"), mp_crop)

        # 7. mediapipe_landmarks_on_mp_crop.jpg
        lm_mp = mp_landmarker.detect(to_mp_image(mp_crop))
        save_image(os.path.join(out_dir, "mediapipe_landmarks_on_mp_crop.jpg"),
                   draw_mediapipe_landmarks(mp_crop, lm_mp))

        cx1, cy1, cx2, cy2 = mp_crop_bbox
        row["mp_landmark_count"] = len(landmarks_to_original(
            lm_mp, cx1, cy1, cx2 - cx1, cy2 - cy1))
    else:
        warn_img  = draw_bbox(image_bgr, (5, 5, max(10, w // 3), max(10, h // 10)),
                              color=(0, 0, 255), label="MediaPipe: no face")
        blank     = blank_image_like(image_bgr, "MediaPipe: no face")
        save_image(os.path.join(out_dir, "mediapipe_bbox.jpg"),  warn_img)
        save_image(os.path.join(out_dir, "mediapipe_crop.jpg"),  blank)
        save_image(os.path.join(out_dir, "mediapipe_landmarks_on_mp_crop.jpg"), blank)

    return row


# ============================================================
# MAIN
# ============================================================
def main():
    ensure_dir(OUTPUT_ROOT)

    # ── discover all clips ─────────────────────────────────────────────────
    print(f"Scanning: {TORONTO_FRAMES_DIR}")
    clips = discover_clips(TORONTO_FRAMES_DIR)
    if not clips:
        raise RuntimeError(f"No clips found under: {TORONTO_FRAMES_DIR}")

    if MAX_CLIPS is not None:
        clips = clips[:MAX_CLIPS]

    total_clips  = len(clips)
    total_frames = sum(len(c["frame_paths"]) for c in clips)
    print(f"Found {total_clips} clips, {total_frames} frames total.")
    print(f"  HC     clips: {sum(1 for c in clips if c['label_str']=='HC')}")
    print(f"  STROKE clips: {sum(1 for c in clips if c['label_str']=='STROKE')}")
    print(f"Output root: {OUTPUT_ROOT}\n")

    # ── build MediaPipe tools once ─────────────────────────────────────────
    print("Loading MediaPipe models...")
    mp_detector   = build_mp_detector()
    mp_landmarker = build_mp_landmarker()
    print("Models loaded.\n")

    # ── manifest rows (one per clip — for your training code) ──────────────
    manifest_rows  = []
    # detailed per-frame summary
    frame_summary  = []

    # ── statistics ─────────────────────────────────────────────────────────
    total_rf_found = 0
    total_mp_found = 0
    total_both_miss = 0
    processed_frames = 0

    for clip_idx, clip in enumerate(clips, start=1):
        label_str = clip["label_str"]
        label_int = clip["label_int"]
        exercise  = clip["exercise"]
        clip_name = clip["clip_name"]
        frames    = clip["frame_paths"]

        # Output folder for this clip's preprocessed crops
        clip_out_root = os.path.join(
            OUTPUT_ROOT, label_str, exercise, clip_name
        )

        print(f"[{clip_idx:04d}/{total_clips}] {label_str}/{exercise}/{clip_name} "
              f"({len(frames)} frames)")

        clip_rf_found = 0
        clip_mp_found = 0

        for frame_path in frames:
            frame_stem = Path(frame_path).stem          # e.g. frame_0001
            frame_out  = os.path.join(clip_out_root, frame_stem)

            row = process_frame(frame_path, frame_out, mp_detector, mp_landmarker)
            if row is None:
                print(f"  [WARN] Could not read: {frame_path}")
                continue

            processed_frames += 1
            if row["retinaface_found"]:
                clip_rf_found += 1
                total_rf_found += 1
            if row["mediapipe_found"]:
                clip_mp_found += 1
                total_mp_found += 1
            if not row["retinaface_found"] and not row["mediapipe_found"]:
                total_both_miss += 1

            row.update({
                "label_str":  label_str,
                "label_int":  label_int,
                "exercise":   exercise,
                "clip_name":  clip_name,
                "frame_stem": frame_stem,
                "frame_path": frame_path,
                "frame_out":  frame_out,
            })
            frame_summary.append(row)

        # ── manifest row for this clip ─────────────────────────────────────
        # Points to the folder of retinaface_crop.jpg files — ready for
        # your SimpleVideoDataset to consume directly.
        manifest_rows.append({
            "clip_dir":       clip_out_root,   # feed this to SimpleVideoDataset
            "label":          label_int,        # 0=HC, 1=STROKE
            "label_str":      label_str,
            "exercise":       exercise,
            "clip_name":      clip_name,
            "total_frames":   len(frames),
            "rf_found":       clip_rf_found,
            "mp_found":       clip_mp_found,
            "rf_detect_rate": clip_rf_found / max(len(frames), 1),
            "mp_detect_rate": clip_mp_found / max(len(frames), 1),
        })

        print(f"  RetinaFace: {clip_rf_found}/{len(frames)} | "
              f"MediaPipe: {clip_mp_found}/{len(frames)}")

    # ── save summary JSON ──────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_ROOT, "frame_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(frame_summary, f, indent=2)

    # ── save manifest CSV — plug directly into your training code ──────────
    import csv
    manifest_path = os.path.join(OUTPUT_ROOT, "manifest_preprocessed.csv")
    if manifest_rows:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)

    # ── final stats ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Clips processed        : {total_clips}")
    print(f"  Frames processed       : {processed_frames}")
    print(f"  RetinaFace detections  : {total_rf_found}/{processed_frames} "
          f"({100*total_rf_found/max(processed_frames,1):.1f}%)")
    print(f"  MediaPipe detections   : {total_mp_found}/{processed_frames} "
          f"({100*total_mp_found/max(processed_frames,1):.1f}%)")
    print(f"  Both missed            : {total_both_miss}/{processed_frames} "
          f"({100*total_both_miss/max(processed_frames,1):.1f}%)")
    print(f"{'='*60}")
    print(f"\n  Output folder  : {OUTPUT_ROOT}")
    print(f"  Frame summary  : {summary_path}")
    print(f"  Manifest CSV   : {manifest_path}")
    print(f"\n  Output structure per frame:")
    print(f"    {{OUTPUT_ROOT}}/{{HC|STROKE}}/{{exercise}}/{{clip_name}}/{{frame_stem}}/")
    print(f"      original.jpg")
    print(f"      retinaface_bbox.jpg")
    print(f"      mediapipe_bbox.jpg")
    print(f"      retinaface_crop.jpg                        <- use for RGB branch")
    print(f"      mediapipe_crop.jpg")
    print(f"      mediapipe_landmarks_on_retina_crop.jpg     <- visual QC")
    print(f"      mediapipe_landmarks_on_mp_crop.jpg         <- visual QC")
    print(f"\n  manifest_preprocessed.csv is ready to use as MANIFEST in your")
    print(f"  training code — clip_dir already points to preprocessed crops.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
