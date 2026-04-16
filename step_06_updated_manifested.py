import pandas as pd
from pathlib import Path

MANIFEST_IN  = r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\Paper_04\FACE_PRECOCESSING\Results\manifest_preprocessed.csv"
MANIFEST_OUT = r"C:\Users\Nicola Sambo\Desktop\Thesis\Data\TORONTO_DATA\Paper_04\FACE_PRECOCESSING\Results\manifest_rf_crops_flat.csv"

df = pd.read_csv(MANIFEST_IN)

# Update clip_dir to point to rf_crops_flat subfolder
df["clip_dir"] = df["clip_dir"].apply(
    lambda p: str(Path(p) / "rf_crops_flat")
)

# Verify all paths actually exist
missing = df[~df["clip_dir"].apply(lambda p: Path(p).exists())]
if len(missing) > 0:
    print(f"WARNING — {len(missing)} clip_dirs do not exist:")
    print(missing[["clip_name", "clip_dir"]].to_string())
else:
    print(f"All {len(df)} clip_dirs verified — paths exist")

df.to_csv(MANIFEST_OUT, index=False)
print(f"Saved: {MANIFEST_OUT}")
print(df[["clip_name", "label_str", "clip_dir"]].head(3).to_string())