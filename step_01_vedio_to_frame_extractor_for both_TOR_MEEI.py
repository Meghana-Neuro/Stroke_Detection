# =============================================================================
# step_01_video_to_frames_FINAL.py
#
# PURPOSE: Extract frames from every video clip in the Toronto NeuroFace
#          dataset (and MEEI) at 10fps, ready for preprocessing.
#hi 
# WHAT CHANGED FROM YOUR ORIGINAL:
#   1. FPS = 5  →  FPS = 10
#      WHY: Short clips (PA=11.9s, BROW=11.7s) at 5fps give only 59 frames.
#           TSN T=16 divides into segments of only 3 frames — almost no
#           temporal diversity. At 10fps same clips give 119 frames → 7 per
#           segment. Clinically: facial muscles move in 100-300ms cycles,
#           well captured at 10fps. At 59.5fps native, 83% of frames are
#           near-identical copies of their neighbour — extracting them all
#           wastes disk and compute with zero clinical benefit.
#
#   2. Added -q:v 2 (near-lossless JPEG)
#      WHY: Default ffmpeg JPEG quality loses fine skin texture details that
#           ResNet18 uses. q:v 2 = near lossless at small file size increase.
#
#   3. Added FFMPEG_EXE detection properly
#      WHY: Your original referenced FFMPEG_EXE but never defined it,
#           which would cause a NameError at runtime.
#
#   4. Added frame count reporting
#      WHY: Tells you immediately if a clip produced the expected frames.
#
# WHAT DID NOT CHANGE:
#   - Folder structure: VIDEO_ROOT / HC|STROKE / EXERCISE / CLIP / VIDEO
#   - Output structure: FRAME_ROOT / HC|STROKE / EXERCISE / CLIP_FOLDER / CLIP_NAME
#   - All video formats supported
#   - Sorted iteration for reproducibility
#
# HOW TO RUN:
#   1. Set VIDEO_ROOT to your raw video folder
#   2. Set FRAME_ROOT to where extracted frames should go
#   3. python step_01_video_to_frames_FINAL.py
#
# FOR MEEI VIDEOS (portrait 1080x1920):
#   Change FPS = 10 stays the same.
#   Add the MEEI_MODE = True flag and set MEEI_VIDEO_ROOT separately.
#   The script handles portrait→landscape rotation automatically.
# =============================================================================

from pathlib import Path
import subprocess
import shutil
import os

# ── FIND FFMPEG ───────────────────────────────────────────────────────────────
FFMPEG_EXE = shutil.which("ffmpeg") or "ffmpeg"
print(f"ffmpeg found at: {FFMPEG_EXE}")

# =============================================================================
# ★ CONFIG — CHANGE ONLY THESE PATHS ★
# =============================================================================
VIDEO_ROOT = Path(r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\paper03\TORONTO_DATA")
FRAME_ROOT = Path(r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\Paper_04\TORONTO_FRAMES_10FPS")

# For MEEI videos (portrait 1080x1920) — set these if using MEEI
MEEI_MODE       = False   # set True when processing MEEI
MEEI_VIDEO_ROOT = Path(r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\MEEI")
MEEI_FRAME_ROOT = Path(r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\MEEI_FRAMES_10FPS")
# =============================================================================

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv"}

# ★ CHANGED: 5 → 10 ★
# Reason: short clips (PA=11.9s) at 5fps → only 3 frames per TSN T=16 segment.
# At 10fps → 7 frames per segment. Sufficient temporal diversity for BiGRU.
FPS = 10


def extract_frames_toronto(video_path: Path, out_dir: Path, fps: int = 10):
    """
    Extract frames from a Toronto NeuroFace video at the given fps.
    Toronto videos are already 640x480 — no resize needed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(out_dir / "frame_%04d.jpg")

    cmd = [
        FFMPEG_EXE,
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",        # ★ ADDED: near-lossless JPEG (preserves skin texture)
        out_pattern
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(f"        [ERROR] ffmpeg failed on {video_path.name}")
        print(f"        {result.stderr[-300:]}")
        return 0

    # Count extracted frames
    n_frames = len(list(out_dir.glob("frame_*.jpg")))
    return n_frames


def extract_frames_meei(video_path: Path, out_dir: Path, fps: int = 10):
    """
    Extract frames from a MEEI video.
    MEEI videos are 1080x1920 portrait — rotate to landscape and resize to 640x480.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(out_dir / "frame_%04d.jpg")

    # transpose=1 rotates 90° clockwise (portrait → landscape)
    # then scale=640:480 normalises to Toronto resolution
    cmd = [
        FFMPEG_EXE,
        "-y",
        "-i", str(video_path),
        "-vf", f"transpose=1,scale=640:480,fps={fps}",
        "-q:v", "2",
        out_pattern
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(f"        [ERROR] {video_path.name}")
        return 0

    return len(list(out_dir.glob("frame_*.jpg")))


def process_toronto():
    """Process all Toronto NeuroFace videos."""
    print(f"\n{'='*60}")
    print(f"TORONTO NEUROFACE — Frame Extraction at {FPS}fps")
    print(f"Video root : {VIDEO_ROOT}")
    print(f"Frame root : {FRAME_ROOT}")
    print(f"{'='*60}")

    if not VIDEO_ROOT.exists():
        raise FileNotFoundError(f"VIDEO_ROOT does not exist: {VIDEO_ROOT}")

    FRAME_ROOT.mkdir(parents=True, exist_ok=True)

    total_clips  = 0
    total_frames = 0

    # Walk: HC/STROKE → EXERCISE → CLIP_FOLDER → VIDEO_FILE
    for class_dir in sorted(VIDEO_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.strip()   # HC or STROKE
        print(f"\n[CLASS] {class_name}")

        for task_dir in sorted(class_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name.strip()
            print(f"  [TASK] {task_name}")

            for clip_dir in sorted(task_dir.iterdir()):
                if not clip_dir.is_dir():
                    continue
                clip_folder_name = clip_dir.name.strip()
                print(f"    [CLIP] {clip_folder_name}")

                for video_path in sorted(clip_dir.iterdir()):
                    if not video_path.is_file():
                        continue
                    if video_path.suffix.lower() not in VIDEO_EXTS:
                        continue

                    clip_name = video_path.stem
                    out_dir = (FRAME_ROOT / class_name / task_name
                               / clip_folder_name / clip_name)

                    n = extract_frames_toronto(video_path, out_dir, FPS)

                    if n > 0:
                        print(f"      [OK] {video_path.name} → {n} frames")
                        total_clips  += 1
                        total_frames += n
                    else:
                        print(f"      [FAIL] {video_path.name}")

    print(f"\n{'='*60}")
    print(f"Toronto done.")
    print(f"  Clips processed : {total_clips}")
    print(f"  Total frames    : {total_frames:,}")
    print(f"  (Your original at 5fps had ~15,522 — at 10fps expect ~31,000)")
    print(f"{'='*60}")


def process_meei():
    """Process all MEEI videos."""
    print(f"\n{'='*60}")
    print(f"MEEI — Frame Extraction at {FPS}fps (portrait→landscape)")
    print(f"Video root : {MEEI_VIDEO_ROOT}")
    print(f"Frame root : {MEEI_FRAME_ROOT}")
    print(f"{'='*60}")

    if not MEEI_VIDEO_ROOT.exists():
        raise FileNotFoundError(f"MEEI_VIDEO_ROOT does not exist: {MEEI_VIDEO_ROOT}")

    MEEI_FRAME_ROOT.mkdir(parents=True, exist_ok=True)

    total_clips  = 0
    total_frames = 0

    # Walk: LABEL / SUBJECT / VIDEO
    for label_dir in sorted(MEEI_VIDEO_ROOT.iterdir()):
        if not label_dir.is_dir():
            continue
        label_name = label_dir.name.strip()
        print(f"\n[LABEL] {label_name}")

        for subject_dir in sorted(label_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject_id = subject_dir.name

            for video_path in sorted(subject_dir.iterdir()):
                if not video_path.is_file():
                    continue
                if video_path.suffix.lower() not in VIDEO_EXTS:
                    continue

                clip_name = video_path.stem
                out_dir   = MEEI_FRAME_ROOT / label_name / subject_id / clip_name

                n = extract_frames_meei(video_path, out_dir, FPS)

                if n > 0:
                    print(f"  [OK] {subject_id}/{video_path.name} → {n} frames")
                    total_clips  += 1
                    total_frames += n
                else:
                    print(f"  [FAIL] {video_path.name}")

    print(f"\nMEEI done. Clips: {total_clips}, Frames: {total_frames:,}")


if __name__ == "__main__":
    if MEEI_MODE:
        process_meei()
    else:
        process_toronto()