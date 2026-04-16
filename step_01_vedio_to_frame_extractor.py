from pathlib import Path
import subprocess
import shutil

# ====== FFMPEG DETECTION (auto-detect or fallback to known path) ======
FFMPEG_EXE = shutil.which("ffmpeg") or r"C:\Users\Nicola Sambo\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"

if not Path(FFMPEG_EXE).exists():
    raise FileNotFoundError(
        f"\n[ERROR] ffmpeg not found at: {FFMPEG_EXE}\n"
        "Please verify the path with:  where ffmpeg"
    )

print(f"ffmpeg found at: {FFMPEG_EXE}")

# ====== CONFIG ======
VIDEO_ROOT = Path(r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\paper03\TORONTO_DATA")
FRAME_ROOT = Path(r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\Paper_04\TORONTO_FRAMES")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv"}
FPS = 10


def extract_frames_ffmpeg(video_path: Path, out_dir: Path, fps: int = 10):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(out_dir / "frame_%04d.jpg")

    cmd = [
        FFMPEG_EXE,
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        out_pattern
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Failed: {video_path}")
        print(result.stderr)
        return False

    n_frames = len(list(out_dir.glob("frame_*.jpg")))
    print(f"      [FRAMES] {n_frames} frames extracted")
    return True


def main():
    print("VIDEO_ROOT:", VIDEO_ROOT)
    print("Exists:", VIDEO_ROOT.exists())
    print(f"FPS: {FPS}")

    if not VIDEO_ROOT.exists():
        raise FileNotFoundError(f"VIDEO_ROOT does not exist: {VIDEO_ROOT}")

    total_videos = 0

    for class_dir in sorted(VIDEO_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.strip()
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
                print(f"    [CLIP_FOLDER] {clip_folder_name}")

                for video_path in sorted(clip_dir.iterdir()):
                    if not video_path.is_file():
                        continue
                    if video_path.suffix.lower() not in VIDEO_EXTS:
                        continue

                    clip_name = video_path.stem
                    out_dir = FRAME_ROOT / class_name / task_name / clip_folder_name / clip_name

                    print(f"      [FOUND] {video_path.name}")
                    ok = extract_frames_ffmpeg(video_path, out_dir, fps=FPS)

                    if ok:
                        print(f"      [OK] -> {out_dir}")
                        total_videos += 1

    print(f"\nDone. Processed {total_videos} videos.")


if __name__ == "__main__":
    main()