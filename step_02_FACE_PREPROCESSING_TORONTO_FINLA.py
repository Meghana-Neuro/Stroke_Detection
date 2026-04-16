import os
import re
import csv
import json
import shutil
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps

from retinaface import RetinaFace
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =============================================================================
# ★ CONFIG — ONLY CHANGE THESE PATHS ★
# =============================================================================
INPUT_FRAMES_DIR = r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\Paper_04\FACE_PRECOCESSING\preprocessing_pipeline\TORONTO_FRAMES_10_FFPS\TORONTO_FRAMES_with_10ffps"
OUTPUT_ROOT      = r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\Paper_04\FACE_PRECOCESSING\preprocessing_pipeline\RESULTS_FROM_PRE_10_ffps"

FACE_LANDMARKER_MODEL_PATH = r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\Paper_04\FACE_PRECOCESSING\face_landmarker.task"
FACE_DETECTOR_MODEL_PATH   = r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\Paper_04\FACE_PRECOCESSING\blaze_face_short_range.tflite"

CLINICAL_CSV = r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\VID_DATASET_Clinical_information_Stroke.csv"
SLP_XLSX     = r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\SLP_Assessment_PS.xlsx"
# =============================================================================

IMAGE_EXTS         = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_MARGIN        = 0.25
MAX_CLIPS          = None   # set to 3 for quick test, then back to None
BLUR_THRESHOLD     = 80
RF_SCORE_THRESHOLD = 0.92
MIN_LANDMARKS      = 400
ROLL_MIN_DEG       = 1.0
ROLL_MAX_DEG       = 30.0


# =============================================================================
# DATASET DISCOVERY
# =============================================================================
def discover_clips(root):
    label_map = {"HC": 0, "STROKE": 1}
    clips     = []
    root_path = Path(root)

    for label_str, label_int in label_map.items():
        label_dir = root_path / label_str
        if not label_dir.exists():
            print(f"  [WARN] folder not found: {label_dir}")
            continue

        for dirpath, dirnames, filenames in os.walk(str(label_dir)):
            dirnames.sort()
            images = sorted([
                os.path.join(dirpath, f)
                for f in filenames
                if Path(f).suffix.lower() in IMAGE_EXTS
            ])
            if not images:
                continue

            dp = Path(dirpath)
            try:
                rel_parts = dp.relative_to(label_dir).parts
                exercise  = rel_parts[0] if rel_parts else "UNKNOWN"
                clip_name = dp.name
            except ValueError:
                exercise  = "UNKNOWN"
                clip_name = dp.name

            m          = re.match(r'^([A-Za-z]+\d+)', clip_name)
            subject_id = m.group(1) if m else "UNKNOWN"

            clips.append({
                "label_str":   label_str,
                "label_int":   label_int,
                "exercise":    exercise,
                "clip_name":   clip_name,
                "subject_id":  subject_id,
                "frame_paths": images,
            })

    clips.sort(key=lambda c: (c["label_str"], c["exercise"], c["clip_name"]))
    print(f"  Discovered {len(clips)} clips")
    return clips


# =============================================================================
# CLINICAL INFO LOADER
# =============================================================================
def load_clinical_info():
    clin, slp = {}, {}
    if CLINICAL_CSV and Path(CLINICAL_CSV).exists():
        try:
            df = pd.read_csv(CLINICAL_CSV)
            for _, row in df.iterrows():
                sid = str(row.get("SubjectID", "")).strip()
                clin[sid] = {
                    "age":              row.get("AgeSession", ""),
                    "gender":           row.get("Gender", ""),
                    "days_from_stroke": row.get("DaysfromStroke", ""),
                    "stroke_type":      row.get("Type", ""),
                    "lesion_location":  row.get("Location", ""),
                    "asymmetry_side":   row.get("Facial asymmetry", ""),
                }
            print(f"  Clinical info: {len(clin)} subjects")
        except Exception as e:
            print(f"  [WARN] Clinical CSV: {e}")
    if SLP_XLSX and Path(SLP_XLSX).exists():
        try:
            df = pd.read_excel(SLP_XLSX)
            for _, row in df.iterrows():
                fname = str(row.get("File Name", "")).strip()
                slp[fname] = {
                    "slp1_total":    row.get("Tot (SLP1)", ""),
                    "slp2_total":    row.get("Tot (SLP2)", ""),
                    "slp1_symmetry": row.get("Symmetry (SLP1)", ""),
                }
            print(f"  SLP scores: {len(slp)} clips")
        except Exception as e:
            print(f"  [WARN] SLP XLSX: {e}")
    return clin, slp


# =============================================================================
# HELPERS
# =============================================================================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_image(path, img):
    cv2.imwrite(path, img)

def blank_image_like(img, msg="No face detected"):
    b = np.zeros_like(img)
    cv2.putText(b, msg, (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,0,255), 2, cv2.LINE_AA)
    return b

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def expand_bbox_xyxy(x1, y1, x2, y2, img_w, img_h, margin=0.25):
    w = x2-x1; h = y2-y1
    mx = int(round(w*margin)); my = int(round(h*margin))
    return (clamp(x1-mx,0,img_w-1), clamp(y1-my,0,img_h-1),
            clamp(x2+mx,0,img_w-1), clamp(y2+my,0,img_h-1))

def crop_from_bbox(img, bbox_xyxy, margin=0.25):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = expand_bbox_xyxy(*bbox_xyxy, w, h, margin)
    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

def draw_bbox(img, bbox_xyxy, color=(0,255,0), label=None, thickness=2):
    out = img.copy()
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    cv2.rectangle(out, (x1,y1), (x2,y2), color, thickness)
    if label:
        cv2.putText(out, label, (x1, max(20,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out

def draw_points(img, points, color=(0,0,255), radius=3):
    out = img.copy()
    for pt in points:
        cv2.circle(out, (int(round(pt[0])),int(round(pt[1]))), radius, color, -1)
    return out

def draw_mediapipe_landmarks(img, lm_result):
    out = img.copy()
    if not lm_result or not getattr(lm_result,"face_landmarks",None): return out
    if not lm_result.face_landmarks: return out
    h, w = out.shape[:2]
    for lm in lm_result.face_landmarks[0]:
        x, y = int(round(lm.x*w)), int(round(lm.y*h))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x,y), 1, (0,255,255), -1)
    return out

def landmarks_to_original(lm_result, cx1, cy1, cw, ch):
    if not lm_result or not getattr(lm_result,"face_landmarks",None): return []
    if not lm_result.face_landmarks: return []
    return [(lm.x*cw+cx1, lm.y*ch+cy1) for lm in lm_result.face_landmarks[0]]


# =============================================================================
# MEDIAPIPE BUILDERS
# =============================================================================
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
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(opts)

def to_mp_image(img):
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# =============================================================================
# RETINAFACE — direct in-memory, NO tempfile
# This is the key fix. The previous version used tempfile for every frame
# which caused Windows file locking → silent exceptions → RF=0 on STROKE.
# =============================================================================
def run_retinaface(image_bgr):
    resp = RetinaFace.detect_faces(image_bgr)
    if not resp or not isinstance(resp, dict): return None
    best_key, best_score, best_area = None, -1.0, -1.0
    for key, info in resp.items():
        fa = info.get("facial_area")
        if not fa or len(fa) != 4: continue
        score = float(info.get("score", -1))
        x1, y1, x2, y2 = fa
        area = max(0, x2-x1) * max(0, y2-y1)
        if score > best_score or (score == best_score and area > best_area):
            best_score, best_area, best_key = score, area, key
    if best_key is None: return None
    info   = resp[best_key]
    bbox   = tuple(map(int, info["facial_area"]))
    lm     = info.get("landmarks", {})
    points = [lm[k] for k in
              ["right_eye","left_eye","nose","mouth_right","mouth_left"]
              if k in lm]
    return {"bbox_xyxy": bbox,
            "score": float(info.get("score", -1)),
            "points": points}

def run_mp_detector(detector, image_bgr):
    result = detector.detect(to_mp_image(image_bgr))
    if not result or not getattr(result,"detections",None) or not result.detections:
        return None
    best_det, best_score, best_area = None, -1.0, -1.0
    for det in result.detections:
        score = float(det.categories[0].score) if det.categories else -1.0
        bb = det.bounding_box
        area = bb.width * bb.height
        if score > best_score or (score == best_score and area > best_area):
            best_score, best_area, best_det = score, area, det
    if best_det is None: return None
    bb   = best_det.bounding_box
    bbox = (int(bb.origin_x), int(bb.origin_y),
            int(bb.origin_x+bb.width), int(bb.origin_y+bb.height))
    h_, w_ = image_bgr.shape[:2]
    points = [(kp.x*w_, kp.y*h_) for kp in (best_det.keypoints or [])]
    return {"bbox_xyxy": bbox,
            "score": float(best_det.categories[0].score)
                     if best_det.categories else -1.0,
            "points": points}


# =============================================================================
# ROLL CORRECTION — safe, no tempfile
# crop_from_bbox(0.25) first → then rotate using eye keypoints → warpAffine
# =============================================================================
def correct_roll(crop_bgr, rf_points_original, crop_x1, crop_y1):
    if not rf_points_original or len(rf_points_original) < 2:
        return crop_bgr.copy(), 0.0, False
    rx = rf_points_original[0][0] - crop_x1
    ry = rf_points_original[0][1] - crop_y1
    lx = rf_points_original[1][0] - crop_x1
    ly = rf_points_original[1][1] - crop_y1
    roll_deg = float(np.degrees(np.arctan2(ry-ly, rx-lx)))
    if abs(roll_deg) < ROLL_MIN_DEG or abs(roll_deg) > ROLL_MAX_DEG:
        return crop_bgr.copy(), roll_deg, False
    h, w = crop_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2.0, h/2.0), roll_deg, 1.0)
    corrected = cv2.warpAffine(crop_bgr, M, (w,h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return corrected, roll_deg, True


# =============================================================================
# QUALITY FILTER
# =============================================================================
def quality_filter(crop_bgr, rf_score, lm_count):
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < BLUR_THRESHOLD:
        return False, f"blur:{lap_var:.1f}", lap_var
    if rf_score < RF_SCORE_THRESHOLD:
        return False, f"low_rf:{rf_score:.3f}", lap_var
    if lm_count < MIN_LANDMARKS:
        return False, f"low_lm:{lm_count}", lap_var
    return True, None, lap_var


# =============================================================================
# ASYMMETRY SCORES
# =============================================================================
def compute_asymmetry_scores(lm_result):
    out = {"eye_asymmetry_score": -1.0, "mouth_asymmetry_score": -1.0}
    if not lm_result or not getattr(lm_result,"face_landmarks",None): return out
    if not lm_result.face_landmarks: return out
    lms = lm_result.face_landmarks[0]
    n   = len(lms)
    def lm(i): return lms[i] if i < n else None
    try:
        Lu,Ld,Li,Lo = lm(159),lm(145),lm(33), lm(133)
        Ru,Rd,Ri,Ro = lm(386),lm(374),lm(362),lm(263)
        if all(x is not None for x in [Lu,Ld,Li,Lo,Ru,Rd,Ri,Ro]):
            ear_L = abs(Lu.y-Ld.y) / (abs(Li.x-Lo.x) + 1e-6)
            ear_R = abs(Ru.y-Rd.y) / (abs(Ri.x-Ro.x) + 1e-6)
            out["eye_asymmetry_score"] = round(abs(ear_L-ear_R), 5)
    except Exception: pass
    try:
        mc_L, mc_R = lm(61), lm(291)
        if mc_L is not None and mc_R is not None:
            out["mouth_asymmetry_score"] = round(abs(mc_L.y-mc_R.y), 5)
    except Exception: pass
    return out


# =============================================================================
# PAD TO SQUARE
# =============================================================================
def pad_to_square(image_bgr):
    h, w = image_bgr.shape[:2]
    if h == w: return image_bgr
    size   = max(h, w)
    pil    = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    padded = ImageOps.pad(pil, (size, size), color=(0,0,0))
    return cv2.cvtColor(np.array(padded), cv2.COLOR_RGB2BGR)


# =============================================================================
# PROCESS ONE FRAME
# Saves 7 files per frame folder + returns row dict for frame_summary.json
# =============================================================================
def process_frame(frame_path, out_dir, mp_detector, mp_landmarker):
    image_bgr = cv2.imread(frame_path)
    if image_bgr is None: return None

    h, w = image_bgr.shape[:2]
    ensure_dir(out_dir)
    save_image(os.path.join(out_dir, "original.jpg"), image_bgr)

    row = {
        "image_w": w, "image_h": h,
        "retinaface_found": False, "mediapipe_found": False,
        "roll_deg": 0.0, "roll_corrected": False,
        "quality_pass": False, "reject_reason": None,
        "laplacian_var": -1.0, "rf_landmark_count": 0,
        "eye_asymmetry_score": -1.0, "mouth_asymmetry_score": -1.0,
    }

    # ── RetinaFace ────────────────────────────────────────────────────────
    retina = run_retinaface(image_bgr)

    if retina is not None:
        row["retinaface_found"] = True
        row["retinaface_score"] = retina["score"]
        x1, y1, x2, y2 = retina["bbox_xyxy"]
        row.update({"rf_x1":x1,"rf_y1":y1,"rf_x2":x2,"rf_y2":y2,
                    "rf_w":x2-x1,"rf_h":y2-y1})

        img_rf = draw_bbox(image_bgr, retina["bbox_xyxy"],
                           color=(0,255,0), label=f"RF {retina['score']:.3f}")
        img_rf = draw_points(img_rf, retina["points"], color=(0,0,255))
        save_image(os.path.join(out_dir, "retinaface_bbox.jpg"), img_rf)

        # crop → roll correct → landmarks → quality → asymmetry → pad → save
        rf_crop, rf_crop_bbox = crop_from_bbox(
            image_bgr, retina["bbox_xyxy"], CROP_MARGIN)
        cx1, cy1, cx2, cy2 = rf_crop_bbox

        rf_crop, roll_deg, roll_corrected = correct_roll(
            rf_crop, retina["points"], cx1, cy1)
        row["roll_deg"]       = round(roll_deg, 2)
        row["roll_corrected"] = roll_corrected

        lm_rf    = mp_landmarker.detect(to_mp_image(rf_crop))
        lm_count = (len(lm_rf.face_landmarks[0])
                    if lm_rf and lm_rf.face_landmarks else 0)
        row["rf_landmark_count"] = lm_count

        save_image(os.path.join(out_dir,
                   "mediapipe_landmarks_on_retina_crop.jpg"),
                   draw_mediapipe_landmarks(rf_crop, lm_rf))

        q_pass, q_reason, lap_var = quality_filter(
            rf_crop, retina["score"], lm_count)
        row["quality_pass"]  = q_pass
        row["reject_reason"] = q_reason
        row["laplacian_var"] = round(lap_var, 2)

        row.update(compute_asymmetry_scores(lm_rf))

        save_image(os.path.join(out_dir, "retinaface_crop.jpg"),
                   pad_to_square(rf_crop))

    else:
        warn  = draw_bbox(image_bgr, (5,5,max(10,w//3),max(10,h//10)),
                          color=(0,0,255), label="RF: no face")
        blank = blank_image_like(image_bgr, "RF: no face")
        save_image(os.path.join(out_dir, "retinaface_bbox.jpg"), warn)
        save_image(os.path.join(out_dir, "retinaface_crop.jpg"), blank)
        save_image(os.path.join(out_dir,
                   "mediapipe_landmarks_on_retina_crop.jpg"), blank)
        row["quality_pass"]  = False
        row["reject_reason"] = "retinaface_missed"

    # ── MediaPipe comparison (QC only — not used for training) ────────────
    mp_det = run_mp_detector(mp_detector, image_bgr)
    if mp_det is not None:
        row["mediapipe_found"] = True
        row["mediapipe_score"] = mp_det["score"]
        x1, y1, x2, y2 = mp_det["bbox_xyxy"]
        row.update({"mp_x1":x1,"mp_y1":y1,"mp_x2":x2,"mp_y2":y2,
                    "mp_w":x2-x1,"mp_h":y2-y1})
        img_mp = draw_bbox(image_bgr, mp_det["bbox_xyxy"],
                           color=(255,0,0), label=f"MP {mp_det['score']:.3f}")
        img_mp = draw_points(img_mp, mp_det["points"], color=(0,255,255))
        save_image(os.path.join(out_dir, "mediapipe_bbox.jpg"), img_mp)
        mp_crop, mp_crop_bbox = crop_from_bbox(
            image_bgr, mp_det["bbox_xyxy"], CROP_MARGIN)
        save_image(os.path.join(out_dir, "mediapipe_crop.jpg"), mp_crop)
        lm_mp = mp_landmarker.detect(to_mp_image(mp_crop))
        save_image(os.path.join(out_dir,
                   "mediapipe_landmarks_on_mp_crop.jpg"),
                   draw_mediapipe_landmarks(mp_crop, lm_mp))
        cx1, cy1, cx2, cy2 = mp_crop_bbox
        row["mp_landmark_count"] = len(landmarks_to_original(
            lm_mp, cx1, cy1, cx2-cx1, cy2-cy1))
    else:
        warn  = draw_bbox(image_bgr, (5,5,max(10,w//3),max(10,h//10)),
                          color=(0,0,255), label="MP: no face")
        blank = blank_image_like(image_bgr, "MP: no face")
        save_image(os.path.join(out_dir, "mediapipe_bbox.jpg"),  warn)
        save_image(os.path.join(out_dir, "mediapipe_crop.jpg"),  blank)
        save_image(os.path.join(out_dir,
                   "mediapipe_landmarks_on_mp_crop.jpg"), blank)

    return row


# =============================================================================
# MAIN
# =============================================================================
def main():
    ensure_dir(OUTPUT_ROOT)
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print(f"  Input : {INPUT_FRAMES_DIR}")
    print(f"  Output: {OUTPUT_ROOT}")
    print("=" * 60)

    print("\nLoading clinical info...")
    clin_dict, slp_dict = load_clinical_info()

    print("\nScanning frames...")
    clips = discover_clips(INPUT_FRAMES_DIR)
    if not clips:
        raise RuntimeError(f"No clips found under: {INPUT_FRAMES_DIR}")
    if MAX_CLIPS is not None:
        clips = clips[:MAX_CLIPS]

    total_clips  = len(clips)
    total_frames = sum(len(c["frame_paths"]) for c in clips)
    print(f"  {total_clips} clips | "
          f"{sum(1 for c in clips if c['label_str']=='HC')} HC | "
          f"{sum(1 for c in clips if c['label_str']=='STROKE')} STROKE | "
          f"{total_frames:,} frames")

    print("\nLoading MediaPipe models...")
    mp_detector   = build_mp_detector()
    mp_landmarker = build_mp_landmarker()
    print("Ready.\n")

    manifest_rows    = []
    frame_summary    = []
    total_rf_found   = 0
    total_mp_found   = 0
    total_q_passed   = 0
    processed_frames = 0

    for clip_idx, clip in enumerate(clips, start=1):
        label_str  = clip["label_str"]
        label_int  = clip["label_int"]
        exercise   = clip["exercise"]
        clip_name  = clip["clip_name"]
        subject_id = clip["subject_id"]
        frames     = clip["frame_paths"]

        clip_out_root = os.path.join(OUTPUT_ROOT, label_str, exercise, clip_name)
        flat_dir      = os.path.join(OUTPUT_ROOT, "rf_crops_flat",
                                     label_str, exercise, clip_name)
        ensure_dir(flat_dir)

        print(f"[{clip_idx:04d}/{total_clips}] "
              f"{label_str}/{exercise}/{clip_name} ({len(frames)} frames)")

        clip_rf_found = 0
        clip_mp_found = 0
        clip_q_passed = 0
        eye_scores, mouth_scores = [], []

        for frame_path in frames:
            frame_stem = Path(frame_path).stem
            frame_out  = os.path.join(clip_out_root, frame_stem)

            row = process_frame(frame_path, frame_out,
                                mp_detector, mp_landmarker)
            if row is None:
                print(f"  [WARN] unreadable: {frame_path}")
                continue

            processed_frames += 1

            if row["retinaface_found"]: clip_rf_found += 1; total_rf_found += 1
            if row["mediapipe_found"]:  clip_mp_found += 1; total_mp_found += 1
            if row["quality_pass"]:     clip_q_passed += 1; total_q_passed += 1

            if row["eye_asymmetry_score"]   >= 0:
                eye_scores.append(row["eye_asymmetry_score"])
            if row["mouth_asymmetry_score"] >= 0:
                mouth_scores.append(row["mouth_asymmetry_score"])

            row.update({
                "label_str": label_str, "label_int": label_int,
                "exercise": exercise,   "clip_name": clip_name,
                "subject_id": subject_id,
                "frame_stem": frame_stem, "frame_path": frame_path,
                "frame_out":  frame_out,
            })
            frame_summary.append(row)

            # Copy to flat_dir ONLY if quality passed
            if row["quality_pass"]:
                src = os.path.join(frame_out, "retinaface_crop.jpg")
                dst = os.path.join(flat_dir, f"{frame_stem}.jpg")
                if os.path.exists(src):
                    shutil.copy2(src, dst)

        mean_eye   = round(float(np.mean(eye_scores)),   5) if eye_scores   else -1.0
        mean_mouth = round(float(np.mean(mouth_scores)), 5) if mouth_scores else -1.0

        print(f"  RF={clip_rf_found}/{len(frames)} | "
              f"quality={clip_q_passed}/{len(frames)} | "
              f"eye={mean_eye:.4f} | mouth={mean_mouth:.4f}")

        c = clin_dict.get(subject_id, {})
        s = slp_dict.get(clip_name + ".avi", slp_dict.get(clip_name, {}))

        manifest_rows.append({
            "clip_dir":             flat_dir,
            "label":                label_int,
            "label_str":            label_str,
            "subject_id":           subject_id,
            "exercise":             exercise,
            "clip_name":            clip_name,
            "total_frames":         len(frames),
            "rf_found":             clip_rf_found,
            "mp_found":             clip_mp_found,
            "quality_passed":       clip_q_passed,
            "rf_detect_rate":       round(clip_rf_found/max(len(frames),1), 4),
            "mp_detect_rate":       round(clip_mp_found/max(len(frames),1), 4),
            "quality_rate":         round(clip_q_passed/max(len(frames),1), 4),
            "mean_eye_asymmetry":   mean_eye,
            "mean_mouth_asymmetry": mean_mouth,
            "age":                  c.get("age",""),
            "gender":               c.get("gender",""),
            "days_from_stroke":     c.get("days_from_stroke",""),
            "stroke_type":          c.get("stroke_type",""),
            "lesion_location":      c.get("lesion_location",""),
            "asymmetry_side":       c.get("asymmetry_side",""),
            "slp1_total":           s.get("slp1_total",""),
            "slp2_total":           s.get("slp2_total",""),
            "slp1_symmetry":        s.get("slp1_symmetry",""),
        })

    # Save frame_summary.json
    summary_path = os.path.join(OUTPUT_ROOT, "frame_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(frame_summary, f, indent=2)

    # Save manifest CSV
    manifest_path = os.path.join(OUTPUT_ROOT, "manifest_preprocessed.csv")
    if manifest_rows:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"\n{'='*60}")
    print("  PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Clips processed  : {total_clips}")
    print(f"  Frames processed : {processed_frames:,}")
    print(f"  RetinaFace found : {total_rf_found:,}/{processed_frames:,} "
          f"({100*total_rf_found/max(processed_frames,1):.1f}%)")
    print(f"  MediaPipe found  : {total_mp_found:,}/{processed_frames:,} "
          f"({100*total_mp_found/max(processed_frames,1):.1f}%)")
    print(f"  Quality passed   : {total_q_passed:,}/{processed_frames:,} "
          f"({100*total_q_passed/max(processed_frames,1):.1f}%)")
    print(f"{'='*60}")
    print(f"  frame_summary.json    : {summary_path}")
    print(f"  manifest_preprocessed : {manifest_path}")
    print(f"  Training crops        : "
          f"{os.path.join(OUTPUT_ROOT, 'rf_crops_flat')}")
    print(f"\n  Per frame folder saves:")
    print(f"    original.jpg")
    print(f"    retinaface_bbox.jpg")
    print(f"    retinaface_crop.jpg          ← training crop")
    print(f"    mediapipe_bbox.jpg")
    print(f"    mediapipe_crop.jpg")
    print(f"    mediapipe_landmarks_on_retina_crop.jpg")
    print(f"    mediapipe_landmarks_on_mp_crop.jpg")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
