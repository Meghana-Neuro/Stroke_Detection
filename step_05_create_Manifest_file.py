from pathlib import Path
import pandas as pd

# ========= CONFIG =========
ROOT = Path(r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\Paper_04\TORONTO_FRAMES")
SAVE_PATH = Path(r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\Paper_04\manifest_face_only.csv")
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    "HC": 0,
    "STROKE": 1,
}

TASKS = {"BBP", "BIG_SMILE", "BLOW", "BROW", "KISS", "OPEN", "PA", "PATAKA", "SPREAD"}

# ========= BUILD MANIFEST =========
rows = []

for class_dir in ROOT.iterdir():
    
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name.strip().upper()
    if class_name not in CLASS_MAP:
        print(f"Skipping unknown class folder: {class_name}")
        continue

    label = CLASS_MAP[class_name]

    for task_dir in class_dir.iterdir():
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name.strip()
        if task_name not in TASKS:
            print(f"Skipping unknown task folder: {task_name}")
            continue

        for clip_folder_dir in task_dir.iterdir():
            if not clip_folder_dir.is_dir():
                continue

            clip_folder_name = clip_folder_dir.name.strip()

            # Subject ID from first token
            subject_id = clip_folder_name.split("_")[0]

            # Inside clip folder, there should be one or more frame folders
            for frame_clip_dir in clip_folder_dir.iterdir():
                if not frame_clip_dir.is_dir():
                    continue

                has_frames = any(
                    p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    for p in frame_clip_dir.iterdir()
                )

                if not has_frames:
                    continue

                rows.append({
                    "clip_dir": str(frame_clip_dir).replace("\\", "/"),
                    "label": label,
                    "subject_id": subject_id,
                    "task": task_name,
                })

df = pd.DataFrame(rows)

if len(df) == 0:
    raise ValueError("No clips found. Please check ROOT path and folder structure.")

df = df.sort_values(["label", "subject_id", "task", "clip_dir"]).reset_index(drop=True)
df.to_csv(SAVE_PATH, index=False)

print(f"\nSaved manifest to: {SAVE_PATH}")
print(f"Total samples: {len(df)}")
print(f"Unique subjects: {df['subject_id'].nunique()}")
print("\nClass counts:")
print(df["label"].value_counts())
print("\nTask counts:")
print(df["task"].value_counts())
print("\nPreview:")
print(df.head())