import os, shutil
from pathlib import Path

RESULTS = r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\Paper_04\FACE_PRECOCESSING\Results"

for frame_dir in Path(RESULTS).rglob("retinaface_crop.jpg"):
    clip_dir  = frame_dir.parent.parent          # e.g. .../N001_02_BBP_NORMAL_color/
    flat_dir  = clip_dir / "rf_crops_flat"
    flat_dir.mkdir(exist_ok=True)
    stem      = frame_dir.parent.name            # e.g. frame_0001
    dest      = flat_dir / f"{stem}.jpg"
    shutil.copy2(frame_dir, dest)

print("Done — rf_crops_flat folders created.")
